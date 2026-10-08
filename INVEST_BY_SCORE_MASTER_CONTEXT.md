# INVEST-BY-SCORE — MASTER PROJECT CONTEXT & ML/FORECASTING ROADMAP

> **Purpose of this document:** This file is a portable context package for AI coding agents such as GitHub Copilot, Cursor, Cline, Claude Code, and similar tools.  
> Read this document **before modifying the repository**. It describes the project's purpose, architectural philosophy, current state, long-term ML/forecasting objective, roadmap, constraints, and implementation principles.

---

# 1. PROJECT IDENTITY

**Repository:** `OrenGolov/invest-by-score`  
**Working branch:** `sprints-vs`

This project is **not merely a stock screener, stock-prediction bot, or collection of AI agents**.

The real product is:

> **A deterministic, point-in-time, evidence-based investment research and decision-governance platform that analyzes companies, markets, events, charts, macro conditions, news and sentiment; produces reproducible investment-quality scores and multi-horizon forecasts; explains the evidence behind those outputs; learns from historical outcomes; and refuses to make a recommendation when evidence, validation, risk, or audit requirements are not satisfied.**

The project must eventually answer two different but related questions:

### A. Investment Quality
> "How strong is this company/investment setup?"

### B. Forecast
> "Given everything legitimately knowable at time T, how is this asset expected to behave over horizon H?"

These are **not the same problem** and should remain conceptually separate.

A company can have:
- excellent long-term fundamentals but poor short-term momentum;
- weak fundamentals but a strong event-driven short-term setup;
- a high quality score but a `NO_TRADE` state due to market stress or risk constraints.

---

# 2. THE CENTRAL PROJECT VISION

The system should continuously transform information into measurable evidence and learn how that evidence historically relates to future market behavior.

Conceptually:

```text
RAW INFORMATION
    |
    +-- Market prices / volume
    +-- Technical observations
    +-- Fundamentals
    +-- News
    +-- Social/sentiment
    +-- Macro releases
    +-- Market regime
    +-- Influential-person statements
    +-- Company events
    +-- Sector / benchmark behavior
    |
    v
POINT-IN-TIME EVIDENCE
    |
    v
FEATURES + EVENTS + MARKET CONTEXT
    |
    +----------------------+---------------------+
    |                      |                     |
    v                      v                     v
INVESTMENT SCORE      EVENT INTELLIGENCE    FORECASTING
    |                      |                     |
    +----------------------+---------------------+
                           |
                           v
                    RISK + AUDITOR
                           |
                    +------+------+
                    |             |
                    v             v
                  PAPER       NO_TRADE
                    |
                    v
               ACTUAL OUTCOME
                    |
                    v
             LEARNING MEMORY
                    |
                    v
             FUTURE MODELS
```

The ultimate system is therefore a **closed research/forecasting loop**:

```text
Observe -> Represent -> Analyze -> Forecast -> Observe Outcome -> Evaluate Error -> Learn -> Validate -> Promote
```

The learning loop must never compromise reproducibility, point-in-time correctness, or governance.

---

# 3. WHY ML IS A MAJOR PART OF THE PROJECT

ML is **not an optional decorative feature**.

The project explicitly aims to learn relationships between:

- information/events;
- chart behavior;
- market regime;
- sector behavior;
- macro conditions;
- company fundamentals;
- sentiment;
- historical analogs;
- future returns and risk.

The system should eventually learn patterns such as:

```text
EVENT
Trump / influential person / company announcement
        +
CURRENT CHART STATE
        +
MARKET REGIME
        +
SECTOR STATE
        +
MACRO CONDITIONS
        +
HISTORICAL ANALOGS
        |
        v
EXPECTED FUTURE RESPONSE
```

Example:

> A major influential person posts about company X.

The system must **not** simply label the post "positive."

It should ask:

1. Who made the statement?
2. What exactly was said?
3. Which company/entity does it affect?
4. Is the information new?
5. How credible/relevant is the source?
6. What market/sector conditions exist?
7. What did the chart look like immediately before the event?
8. What happened historically after similar events?
9. What happened to the benchmark and sector at the same time?
10. What is the stock's expected behavior over 1D / 5D / 20D / 60D / longer horizons?
11. How confident is the model?
12. Is the prediction calibrated?
13. Is there a risk/audit veto?

The goal is **not causal certainty from observational data**. The system must distinguish event-associated reaction from proven causality.

---

# 4. CHART ANALYSIS IS A CORE COMPETENCY

A critical project requirement is:

> **Understand how information and events affect the chart.**

Therefore, chart analysis cannot be treated as a simple technical-score generator.

The system should continuously analyze:

### Intraday
- 5m
- 15m
- 30m
- 1h

### Daily
- 1D

### Longer-term
- weekly
- monthly
- yearly

The exact supported intervals may evolve, but the architecture must support multi-timeframe analysis.

The system should learn relationships between:

```text
EVENT
    ->
pre-event chart state
    ->
immediate reaction
    ->
1D reaction
    ->
5D reaction
    ->
20D reaction
    ->
60D reaction
    ->
long-term consequence
```

Chart features should eventually include:

- returns
- momentum
- acceleration
- volatility
- ATR
- trend slope
- volume surprise
- gap behavior
- drawdown
- recovery speed
- support/resistance distance
- breakout state
- failed breakout
- volatility expansion
- volatility contraction
- higher-high / higher-low structure
- lower-high / lower-low structure
- relative strength
- benchmark-relative strength
- sector-relative strength

Do not add features merely because they can be calculated. Every production feature must be registered, versioned, tested, PIT-safe and owned by a contract.

---

# 5. MARKET CONTEXT IS REQUIRED

A stock cannot be interpreted in isolation.

Forecasts should eventually include:

```text
S&P 500
Nasdaq
Russell 2000
VIX
10Y Treasury yield
USD
relevant sector ETF
relevant industry benchmark
```

At minimum, the system should be able to distinguish:

```text
Stock +4%
while S&P +4%
```

from:

```text
Stock +4%
while S&P -2%
```

The latter is much stronger relative relative-strength evidence.

Market context must be point-in-time and versioned.

---

# 6. POINT-IN-TIME CORRECTNESS — NON-NEGOTIABLE

The project is fundamentally a **point-in-time research system**.

For any:

```text
ticker + as_of timestamp
```

the system may only use information that was legitimately available at that timestamp.

This applies to:

- prices
- technical indicators
- fundamentals
- SEC/financial reports
- news
- social posts
- macro releases
- macro revisions
- index constituents
- corporate actions
- model inputs
- labels
- event metadata

### Critical example

A macro value may describe July but be published on August 12.

A prediction made on August 5 **must not use the August 12 publication**.

Publication time matters.

### Historical revision rule

When possible, retain the first-publication vintage.

Never silently replace historical knowledge with later revised knowledge in a historical forecast.

---

# 7. EVIDENCE AND PROVENANCE

Every meaningful analytical result should be traceable to evidence.

The system should be able to answer:

> "Why did you produce this score/forecast at this exact time?"

Evidence should include:

- source
- publication time
- effective time where applicable
- retrieval time
- source quality
- entity
- event type
- feature version
- model version
- relevant data digest/hash
- provenance metadata

Historical outputs must remain immutable.

---

# 8. AGENT ARCHITECTURE

The project uses specialized agents rather than a single monolithic decision maker.

The intended decision roster is:

1. Market Data
2. Technical
3. Fundamental
4. News
5. Sentiment
6. Macro
7. Regime
8. Risk
9. Auditor

The important principle is:

> **Every agent is born wired.**

A new agent must not merely exist as a class/module. Its output must have a real, governed path into the orchestration architecture.

The system should know whether an agent is:

- `OK`
- `UNAVAILABLE`
- `STALE`
- `INCOMPLETE`
- `CONTRADICTORY`
- `INVALID`

Agent status is part of the decision state.

---

# 9. GOVERNANCE PRINCIPLES

The system is **fail-closed**.

It must never silently convert:

- missing data -> neutral evidence
- stale data -> valid evidence
- contradictory evidence -> average/neutral evidence
- invalid output -> usable output
- unavailable provider -> fabricated data

into a positive decision.

Possible final posture includes:

```text
PAPER
ANALYSIS_ONLY
NO_TRADE
```

A high score does not override a risk or audit veto.

---

# 10. RISK AND AUDITOR ARE REAL GATES

Risk and Auditor are not informational sidebars.

### Risk

Must be capable of producing a veto.

Risk rules should be explicit and versioned.

A triggered veto-severity rule must make it impossible for the system to produce an executable/paper action when the policy forbids it.

### Auditor

Must validate things such as:

- evidence sufficiency
- hash integrity
- deterministic replay
- calibration sanity
- decision reproducibility

An auditor veto must propagate to the final posture.

---

# 11. ENSEMBLE DESIGN

The investment-quality score is a governed ensemble.

It must support versioned weight sets, e.g.:

```text
CURRENT
LONG
```

Weights must sum to exactly 1.0.

If some agents are non-OK, weights may be renormalized over eligible agents according to explicit rules.

If no eligible agents remain:

```text
NO_TRADE
```

The output must expose an `ensemble_breakdown`.

The system must avoid split-brain scoring.

There should be one canonical technical scoring implementation; wrappers should delegate to it.

---

# 12. RAW DATA AND REPLAYABILITY

The project uses an append-only raw-record concept.

Raw data should support:

```text
data/raw/.../*.jsonl
```

with:

- SHA-256 integrity
- immutable records
- supersede-on-refetch semantics
- reproducible rebuild

The system should be able to demonstrate:

> Delete derived cache -> rebuild derived data from raw -> reproduce the same result.

This is essential for a serious research platform.

---

# 13. SPRINT HISTORY / PROJECT STORY

The project evolved in the following conceptual sequence:

## Early foundation
Architecture and planning established the central principles:
- point-in-time correctness
- deterministic scoring
- evidence
- governance
- replayability
- separation of analysis from execution

## Sprint 1–2
A working scoring engine was created.

This established the first actual score while exposing the need for stronger governance.

## Sprint 3
Fundamental scoring contracts were introduced.

## Sprint 4
Real valuation/fundamental data and the local application were added.

## Later work
The project progressively added:
- typed agent/orchestrator contracts
- governance hardening
- provider fallback
- audit persistence
- score metadata
- provider health
- replay lookup
- canonical `MarketSnapshot`
- evidence-based confidence

Important architectural fixes established the principle that bugs in scoring must be fixed together with regression tests.

## W — Governance Wiring
W exists to make the multi-agent architecture genuinely govern the score.

Tasks:
- W1 versioned ensemble
- W2 fail-closed risk gate
- W3 auditor as evidence/replay validator
- W4 agent status taxonomy
- W5 one technical truth
- W6 append-only raw store
- W7 audit-event enrichment

W is not "another feature sprint"; it is the governance nervous system.

---

# 14. CURRENT POSITION

The project is now moving into:

# Sprint N — News / Sentiment / Macro / Regime

The previous draft defined 6 tasks:

- news adapter + classifier taxonomy + contradiction handling
- sentiment placeholder + anti-proxying rule
- macro series registry + publication-time vintages
- 5-state regime classifier + stress -> NO_TRADE coupling
- narrative-vs-fundamental attribution
- slotted feature gaps such as ATR, slope and breadth, deferred until properly wired

This is the correct immediate direction.

But N should be understood as **contextual intelligence**, not simply "add a news agent."

---

# 15. NEWS AGENT REQUIREMENTS

News processing should become a structured event pipeline:

```text
NEWS
 |
 v
PIT FILTER
 |
 v
ENTITY RESOLUTION
 |
 v
EVENT CLASSIFICATION
 |
 v
SOURCE QUALITY
 |
 v
RELEVANCE / NOVELTY
 |
 v
DIRECTION / MAGNITUDE
 |
 v
CONTRADICTION DETECTION
 |
 v
EVIDENCE-BACKED OUTPUT
```

Taxonomy should include categories such as:

- earnings
- guidance
- regulation
- litigation
- product launch
- M&A
- macro shock
- strategic announcement
- management commentary
- other

Contradictory credible evidence should produce:

```text
CONTRADICTORY
```

rather than silently averaging to neutral.

---

# 16. SENTIMENT REQUIREMENTS

Sentiment must not be fabricated.

If a legitimate sentiment source is unavailable:

```text
UNAVAILABLE
```

Do not infer sentiment from:
- RSI
- price direction
- technical indicators
- generic news score

unless explicitly designed and documented as a derived feature.

If news-derived sentiment is used, it must be explicitly labelled, for example:

```text
derived_from_news
```

and confidence should reflect that dependency.

---

# 17. MACRO REQUIREMENTS

Create a macro series registry.

Each series should have:

- series identifier
- source
- publication timestamp
- reference period
- first-release value
- revision history where available
- frequency
- unit
- transformation
- PIT policy
- feature version

Macro data must be vintage-aware.

### Registered series (N3, macro-adapter-v3)

```text
fed_funds        FEDFUNDS   daily      rates level
cpi_yoy          CPIAUCSL   monthly    inflation
initial_claims   ICSA       weekly     labor
gdp_growth       A191RL...  quarterly  growth
10y_yield        DGS10      daily      long rate
30y_yield        DGS30      daily      ultra-long rate / term premium
vix              VIXCLS     daily      implied volatility (INVERTED)
```

**VIX is the one inverted series.** Every other macro signal reads
"higher = more risk-on"; a high VIX is the market pricing fear, so its
contribution sign is flipped. A sign error there would turn a panic into a
buy signal.

The regime score is the **mean of available signals**, so widening the
series set changes every historical score. Adding a series is therefore a
versioned change (`MACRO_ADAPTER_VERSION`), never a silent one.

---

# 18. REGIME REQUIREMENTS

Use five states:

```text
BULLISH
BEARISH
RANGE
RISK_OFF
STRESS
```

Regime should be derived from explicit, versioned inputs.

`STRESS` must couple to:

```text
NO_TRADE
```

according to governance policy.

Regime is not merely descriptive; it affects decision posture and eventually forecast conditioning.

---

# 19. NARRATIVE VS FUNDAMENTAL ATTRIBUTION

The system should distinguish:

### Fundamental/business strength
from

### Market narrative strength

Example:

```text
Investment Quality: 8.3

Fundamental contribution: +2.4
Technical contribution: +1.8
News/narrative contribution: +1.2
Macro contribution: -0.3
Regime contribution: +0.7
```

This lets the system explain whether the investment thesis is supported by business reality, market narrative, or both.

---

# 20. MASTER ROADMAP

The previous ML roadmap was too small for the actual project vision.

The recommended master roadmap is:

```text
W  -> Governance Wiring
N  -> News / Sentiment / Macro / Regime
V  -> Labels / Backtest / Paper
M  -> ML Foundation / Model Governance
E  -> Event Intelligence / Market Reaction Learning
C  -> Chart / Temporal Intelligence
F  -> Multi-Horizon Forecasting
L  -> Continuous Learning / Institutional Memory
R  -> Portfolio-Aware Risk
D  -> Research / Forecast Dashboard
A  -> Forecast Alerts
X  -> Scientific Release Gate
```

Do not collapse these into a single "ML sprint."

---

# 21. SPRINT M — ML FOUNDATION & MODEL REGISTRY

## M1 — Canonical Feature Registry

Every ML feature must have:
- name
- owner
- domain
- formula
- version
- unit
- frequency
- lookback
- minimum history
- null policy
- PIT rule
- source dependencies
- feature family
- model compatibility

Acceptance:
- unregistered feature cannot enter production model
- producer must exist
- future/revised input rejected
- deterministic feature hash

## M2 — Official Training Dataset Builder

Build the only official dataset generator.

For every training row:

```text
prediction_time
    ->
information available at prediction_time
    ->
features
    ->
future outcome
```

Dataset must be deterministically hashed.

## M3 — Research Trial Registry

Record:
- trial ID
- hypothesis
- feature-set version
- model family
- hyperparameters
- label version
- horizons
- training window
- validation scheme
- costs
- seed
- dataset hash
- metrics

This protects against uncontrolled experimentation and accidental cherry-picking.

## M4 — Baseline Model Suite

Establish strong simple baselines before deep models:
- historical mean
- momentum
- mean reversion
- linear/Ridge
- logistic regression
- tree/boosting methods

No advanced model should be promoted without beating the incumbent OOS.

## M5 — Model Artifact Registry

Store:
- model version
- feature-set version
- training cutoff
- dataset hash
- code commit
- hyperparameters
- seed
- artifact hash
- calibration version
- metrics
- lifecycle status
- parent model
- approval
- forecast target
- horizon
- universe

## M6 — Calibration and Uncertainty

Support:
- probability calibration
- reliability curves
- Brier score
- log loss
- calibration error
- prediction intervals
- uncertainty
- fold dispersion

Never expose arbitrary probability numbers as if they were calibrated.

## M7 — Champion / Challenger

Use:

```text
CHAMPION
  |
  +-- CHALLENGER A
  +-- CHALLENGER B
  +-- CHALLENGER C
```

New models begin in `SHADOW`.

## M8 — ML Reproducibility Gate

Same:
- data
- features
- seed
- code
- configuration

must produce identical model artifacts/predictions within explicitly defined reproducibility guarantees.

---

# 22. SPRINT E — EVENT INTELLIGENCE & MARKET REACTION

This sprint is essential because the project must learn how events affect charts.

## E1 — Canonical Event Object

Schema should include:

```text
event_id
published_time
effective_time
entity
entities_affected
actor
actor_type
event_type
source
source_quality
novelty
relevance
direction
magnitude
confidence
evidence
```

## E2 — Entity Resolution

Resolve:
- company names
- tickers
- executives
- influential people
- organizations
- aliases

Bad entity resolution must not silently enter training.

## E3 — Influential Person Intelligence

Track:
- identity
- role
- organization
- historical relevance
- topic specialization
- source credibility
- statement frequency
- novelty
- historical market impact

Eventually learn:

```text
actor × topic × company/sector
    ->
historical market response
```

## E4 — Event Study Engine

For every event:

```text
event
 ->
pre-event baseline
 ->
stock reaction
 ->
benchmark reaction
 ->
sector reaction
 ->
abnormal return
 ->
volatility response
 ->
volume response
```

Measure:
- intraday
- 1D
- 5D
- 20D
- 60D

## E5 — Confounder / Attribution Engine

Never automatically claim causality.

Decompose observed movement conceptually into:

```text
market component
+ sector component
+ stock-specific component
+ event-associated residual
```

Report event-associated impact, not unsupported causal certainty.

## E6 — Event Memory

Store event/reaction examples:

```text
event
current context
chart state
historical analogs
1D response
5D response
20D response
60D response
```

This becomes institutional memory.

## E7 — Event Revision / Contradiction Learning

Store event chains such as:

```text
initial claim
 ->
correction
 ->
confirmation
 ->
reversal
```

The evolution itself can become training information.

---

# 23. SPRINT C — CHART & TEMPORAL INTELLIGENCE

## C1 — Multi-Timeframe Representation

Represent:
- intraday state
- daily state
- weekly trend
- monthly structure
- yearly context

all aligned to the same `as_of`.

## C2 — Price/Volume Feature Expansion

Registered features:
- returns
- momentum
- acceleration
- volatility
- ATR
- volume surprise
- gaps
- slope
- drawdown
- recovery speed
- support/resistance distance
- breakout/failure state
- volatility expansion/contraction
- relative strength

## C3 — Market Context Features

Include:
- S&P 500
- Nasdaq
- Russell 2000
- VIX
- 10Y yield
- USD
- sector ETF
- industry benchmark

## C4 — Deterministic Chart Structure Representation

Represent:
- trend
- consolidation
- breakout
- failed breakout
- gap
- reversal
- volatility expansion
- volatility contraction
- higher-high/higher-low
- lower-high/lower-low

Avoid beginning with screenshot interpretation or vague pattern labels.

## C5 — Temporal Sequence Dataset

Create point-in-time sequences:

```text
T-60
...
T-1
T0
```

with:
- price
- volume
- technical state
- market state
- sector state
- events
- macro
- sentiment
- fundamentals

## C6 — Sequence Model Research

Evaluate only after baseline validation:
- temporal convolution
- LSTM/GRU
- transformer/time-series transformer
- temporal fusion approaches

Complexity must earn its place through OOS evidence.

## C7 — Chart Reaction Memory

For significant events, retain:

```text
before event
event
1h reaction
1D reaction
5D reaction
20D reaction
60D reaction
```

This supports historical analog retrieval.

---

# 24. SPRINT F — MULTI-HORIZON FORECASTING ENGINE

This is the central forecasting product.

## F1 — Forecast Targets

Support separate targets:

### Direction
```text
P(return > 0)
```

### Expected return
```text
E(return)
```

### Return distribution
```text
P(return in interval)
```

### Downside / adverse excursion
```text
expected downside
```

### Volatility
```text
expected future volatility
```

### Relative performance
```text
P(stock > benchmark)
```

## F2 — Horizons

At minimum:

```text
1D
5D
20D
60D
120D
252D
```

The exact horizon set can evolve.

## F3 — Joint Forecast

Example:

```text
1D:   +0.8%   P(up)=63%
5D:   +2.1%   P(up)=67%
20D:  +4.7%   P(up)=72%
60D:  +8.9%   P(up)=69%
```

with calibrated uncertainty.

## F4 — Conditional Forecasting

Forecasts should condition on context:

```text
P(+5% in 20D | bullish regime)
```

versus:

```text
P(+5% in 20D | stress regime)
```

## F5 — Event-Conditioned Forecast

Pipeline:

```text
new event
 ->
event representation
 ->
historical event matches
 ->
current chart
 ->
market regime
 ->
forecast
```

## F6 — Forecast Decomposition

Example:

```text
20D forecast: +6.2%

Technical:        +2.1%
Fundamental:      +1.4%
News/Event:       +1.8%
Macro:            -0.4%
Regime:           +0.9%
Sentiment:        +0.2%
Historical analog:+0.6%
```

Contributions must be interpretable and must not imply false causal certainty.

## F7 — Forecast Confidence

Confidence should account for:
- sample size
- calibration
- model agreement
- feature completeness
- source quality
- regime similarity
- event similarity
- model drift
- uncertainty

`confidence` is not the same as `P(up)`.

## F8 — Forecast API Contract

Create a versioned `ForecastSnapshot` containing at minimum:

```text
ticker
as_of
forecast_version
model_versions
horizon
expected_return
probability_up
prediction_interval
confidence
regime
event_context
feature_digest
evidence
model_contributions
warnings
```

---

# 25. SPRINT L — CONTINUOUS LEARNING & INSTITUTIONAL MEMORY

The phrase "continuous learning" must mean controlled learning, not uncontrolled self-modification.

## L1 — Outcome Closure

Every forecast eventually becomes:

```text
forecast
 ->
horizon expires
 ->
actual outcome
 ->
error
 ->
calibration evaluation
```

## L2 — Forecast Performance Ledger

Measure by:
- ticker
- sector
- regime
- horizon
- event type
- source
- model
- confidence bucket
- volatility regime

## L3 — Error Memory

Store:

```text
forecast
actual
error
context
```

and identify systematic errors.

## L4 — Source Reliability Learning

Source quality may depend on:

```text
source
× event type
× sector
× horizon
```

Do not assume one global source score is sufficient.

## L5 — Controlled Incremental Learning

Never automatically replace the production champion every day.

Correct pipeline:

```text
new data
 ->
candidate update
 ->
shadow evaluation
 ->
drift testing
 ->
OOS validation
 ->
promotion gate
 ->
human approval
 ->
new champion
```

## L6 — Concept Drift Detection

Detect:
- feature distribution drift
- relationship drift
- calibration drift
- event-response drift

## L7 — Regime-Specific Learning

Potentially evaluate separate models for:
- bullish
- bearish
- range
- risk-off
- stress

Only use specialization if OOS evidence supports it.

## L8 — Champion Evolution

Replacement requires:
- credible OOS improvement
- acceptable calibration
- no unacceptable false-positive degradation
- acceptable risk
- regime robustness
- reproducibility
- governance approval

Historical forecasts remain immutable.

---

# 26. SPRINT R — PORTFOLIO-AWARE FORECASTING

Eventually forecasting must understand the portfolio.

Tasks:
- R1 position exposure
- R2 correlation-aware sizing
- R3 sector concentration
- R4 forecast-adjusted risk
- R5 expected portfolio impact
- R6 stress scenarios
- R7 portfolio-level NO_TRADE

The question evolves from:

> "Is NVDA attractive?"

to:

> "Does acting on the NVDA forecast improve the current portfolio without violating risk constraints?"

---

# 27. SPRINT D — RESEARCH / FORECAST DASHBOARD

The dashboard should become a research workstation.

For a stock:

```text
NVDA

Investment Quality     8.6 / 10

Forecast:
1D                     +0.8%   63%
5D                     +2.7%   69%
20D                    +7.2%   74%
60D                   +11.8%   71%

Confidence              0.83
Regime                  BULLISH
Risk                    ACCEPTABLE
```

Then show:

```text
WHY?

Technical              +2.1
Fundamental            +1.7
News                   +1.3
Macro                  -0.4
Regime                 +0.8
Sentiment              +0.2
```

And event context:

```text
Recent event:
Influential person / CEO statement

Classification:
Strategic / Company Outlook

Historical analogs:
42

Median historical response:
+6.1% / 20D

Current setup similarity:
81%
```

The UI must expose provenance, not hide it.

---

# 28. SPRINT A — FORECAST ALERTS

Future alerts should be forecast-driven, not simplistic score alerts.

Examples:

## A1 — Forecast Change
Alert when forecast materially changes.

## A2 — Confidence Change
Alert when confidence changes materially.

## A3 — High-Impact Event
Alert when a high-impact event is detected.

## A4 — Regime Change
Example:

```text
BULLISH -> RISK_OFF
```

## A5 — Thesis Break
Alert when evidence materially contradicts the existing thesis.

## A6 — Forecast Threshold
Example:

```text
20D expected return: +8.3%
P(up): 74%
confidence: 0.82
no veto
```

## A7 — Alert Suppression
Prevent repeated/spam alerts.

Alerts must respect risk and governance.

---

# 29. SPRINT X — SCIENTIFIC RELEASE GATE

Before a model/forecasting release becomes production-capable:

## X1 — OOS forecast validation

Demonstrate performance on unseen data.

## X2 — Calibration gate

Probabilities must be demonstrably calibrated.

## X3 — Regime robustness

No single regime should explain the entire edge.

## X4 — Event robustness

No single viral event/source should explain the apparent edge.

## X5 — Feature ablation

Test whether removing:
- news
- technicals
- macro
- sentiment
- fundamentals

actually changes performance.

This proves which information sources provide incremental value.

## X6 — Temporal robustness

Evaluate multiple horizons:
- 1D
- 5D
- 20D
- 60D
- 120D

## X7 — Multiple-testing protection

Track experiments and guard against selection bias.

Potential methods:
- bootstrap/permutation approaches
- Reality Check / SPA-style methods
- Probability of Backtest Overfitting
- Deflated Sharpe where applicable

Use methods appropriate to the target/metric.

## X8 — Sealed holdout

Maintain an untouched final period.

Once opened:

**do not use it for further model selection.**

## X9 — Release snapshot

Freeze:
- code
- data
- features
- models
- weights
- calibration
- configuration
- validation results

## X10 — Honest gate report

If evidence is insufficient:

```text
FORECASTING RELEASE: NOT APPROVED

Reason:
OOS directional edge not statistically established.

Action:
Continue shadow evaluation.
```

"Gate not met" is a valid successful engineering outcome.

---

# 30. IMPORTANT ML DESIGN PRINCIPLE

Do not make the ML model simply replace the 0–10 score.

Maintain two distinct concepts:

## Investment Quality Score

Answers:

> "How strong is the investment/business setup?"

## Forecast Engine

Answers:

> "What is the expected future behavior of this asset over horizon H?"

Then combine them at the governance/decision layer.

Example:

```text
NVDA

Investment Quality: 8.6

Forecast:
1D   +0.4%
5D   +1.9%
20D  +7.4%
60D +11.2%

Confidence: 0.83
Regime: Bullish
Risk: Acceptable
```

The system can then determine:

```text
PAPER
ANALYSIS_ONLY
NO_TRADE
```

without conflating company quality with short-term forecast.

---

# 31. EVENT-TO-CHART LEARNING MODEL

This is one of the project's most important future capabilities.

Target pipeline:

```text
EVENT
  |
  +-- actor
  +-- topic
  +-- polarity
  +-- novelty
  +-- relevance
  +-- source quality
  |
  v
CURRENT MARKET STATE
  |
  +-- S&P
  +-- Nasdaq
  +-- VIX
  +-- sector
  +-- rates
  +-- USD
  |
  v
CURRENT CHART STATE
  |
  +-- trend
  +-- momentum
  +-- volatility
  +-- volume
  +-- structure
  |
  v
HISTORICAL ANALOG RETRIEVAL
  |
  v
CONDITIONAL FORECAST
  |
  +-- 1D
  +-- 5D
  +-- 20D
  +-- 60D
  +-- 120D
  |
  v
CALIBRATION
  |
  v
RISK + AUDIT
  |
  v
FORECAST
  |
  v
ACTUAL OUTCOME
  |
  v
LEARNING MEMORY
```

This is the conceptual core of the future forecasting engine.

---

# 32. WHAT NOT TO DO

AI coding agents must avoid these failure modes.

### Do not:
- add ML before the dataset/label/backtest foundation is trustworthy;
- use future information;
- use revised macro data in historical snapshots without PIT handling;
- infer sentiment from price unless explicitly defined as derived;
- treat news polarity as causality;
- claim an influential person caused a price move without appropriate attribution methodology;
- silently replace missing values with neutral evidence;
- allow a failed agent to masquerade as `OK`;
- allow risk/audit vetoes to be overridden by a high score;
- create duplicate technical scoring implementations;
- let models consume unregistered features;
- let experiments bypass the trial registry;
- automatically promote a newly trained model;
- let continuous learning silently mutate the production champion;
- optimize on the final holdout;
- build a complex neural network before proving simple baselines;
- use screenshots/chart patterns as vague "AI intuition" without measurable representation;
- add indicators just because they look useful;
- make the dashboard hide uncertainty or provenance.

---

# 33. ENGINEERING QUALITY STANDARD

Every implementation should be treated as if reviewed by:

- senior quantitative researchers
- capital-market professionals with 25+ years of experience
- senior software engineers
- senior ML/AI engineers
- data engineers
- risk/governance specialists

For every task:

1. Define the contract.
2. Define inputs/outputs.
3. Define PIT behavior.
4. Define failure states.
5. Define provenance.
6. Define versioning.
7. Define deterministic behavior.
8. Implement.
9. Add regression/unit/integration tests.
10. Verify existing tests.
11. Verify no architectural invariant was broken.
12. Document the behavior.

Do not optimize for "task completed."

Optimize for:

> **correctness + reproducibility + explainability + scientific validity + maintainability.**

---

# 34. TESTING PHILOSOPHY

Every important bug should lead to:

```text
BUG FIX
+
REGRESSION TEST
```

The project's established "house rule" is that fixes must be accompanied by tests that prevent recurrence.

Tests should cover:
- PIT boundaries
- timestamp ordering
- missing data
- stale data
- contradictory evidence
- invalid agent output
- deterministic replay
- hash integrity
- weight normalization
- risk vetoes
- auditor vetoes
- model reproducibility
- label leakage
- train/validation separation
- embargo
- cost assumptions
- event attribution
- calibration
- drift

---

# 35. BACKTESTING PRINCIPLES

Before ML promotion:

- labels must be leakage-safe;
- future information must be excluded;
- walk-forward validation must be used;
- embargo should be used where required;
- transaction costs must be explicit;
- slippage assumptions must be explicit;
- run manifests must be mandatory;
- the same feature/scoring contracts should be used in research and production;
- historical universe construction must avoid survivorship bias;
- corporate actions must be handled correctly.

The backtest is not proof that the future will behave the same way.

It is evidence about historical behavior under explicit assumptions.

---

# 36. FORECASTING OUTPUT PRINCIPLES

Never output:

```text
NVDA will rise 74%.
```

Prefer:

```text
20D forecast:

Expected return: +7.4%
P(return > 0): 74%
Prediction interval: [appropriate calibrated range]
Confidence: 0.83
Model: forecast-vX
Regime: Bullish
Data completeness: ...
Warnings: ...
```

The forecast must communicate uncertainty.

---

# 37. CAUSALITY VS ASSOCIATION

The system is primarily observational.

Therefore:

```text
Event -> price movement
```

does not automatically prove:

```text
Event caused price movement.
```

Use careful terminology:

- event-associated return
- abnormal return
- residual movement
- historical response
- conditional relationship

Only use stronger causal language if a valid causal methodology supports it.

---

# 38. DATA GROWTH / LEARNING PHILOSOPHY

The intended long-term loop is:

```text
More historical data
        +
More event data
        +
More chart observations
        +
More outcomes
        +
Better feature representations
        +
Better validation
        |
        v
Potentially better forecasts
```

But:

> **More data does not automatically mean more accuracy.**

The project must continuously test whether new data/features/models produce genuine OOS improvement.

The system should be capable of learning that a new information source adds **no predictive value**.

That is a valid scientific result.

---

# 39. FUTURE ALERT PRODUCT

The eventual alert system may generate:

### BUY-SETUP / POSITIVE FORECAST
when:
- forecast is sufficiently positive;
- confidence is sufficiently high;
- model is validated;
- evidence is adequate;
- no risk/audit veto exists.

### SELL/RISK ALERT
when:
- forecast deteriorates materially;
- downside probability rises;
- thesis-breaking evidence appears;
- regime deteriorates;
- risk threshold is breached.

### IMPORTANT UPDATE
when:
- a high-impact event arrives;
- forecast changes materially;
- a significant regime transition occurs;
- historical analogs indicate an unusual expected reaction.

These are future product capabilities and should not be implemented prematurely without the forecasting validation layer.

---

# 40. RECOMMENDED EXECUTION ORDER

The intended order is:

```text
W
 |
 v
N
 |
 v
V
 |
 v
M
 |
 v
E
 |
 v
C
 |
 v
F
 |
 v
L
 |
 v
R
 |
 v
D
 |
 v
A
 |
 v
X
```

However, implementation can overlap where dependencies are clean.

The key rule is:

> **Validation infrastructure must precede serious ML promotion.**

Advanced ML research can begin experimentally when the necessary data/labels exist, but production promotion must remain behind the validation gates.

---

# 41. CURRENT AI-CODING-AGENT INSTRUCTIONS

When asked to implement a task in this repository:

### First
Read:
- `README.md`
- `docs/architecture.md`
- `docs/data-model.md`
- `docs/agents.md`
- `docs/features-and-models.md`
- `docs/validation.md`
- `docs/roadmap.md`
- `docs/task-breakdown.md`

Then inspect the relevant implementation and tests.

### Second
Determine:
- current sprint
- existing contract
- dependencies
- existing invariants
- relevant versions
- existing tests
- whether a similar implementation already exists

### Third
Before coding:
- identify the exact files that should change;
- identify potential architectural side effects;
- define acceptance criteria;
- define tests.

### Fourth
Implement the smallest correct change that fully satisfies the task.

Do not rewrite unrelated parts of the project.

### Fifth
Run relevant tests.

If tests fail:
- diagnose root cause;
- fix;
- add regression coverage where appropriate.

### Sixth
Report:
- files changed
- behavior added
- tests executed
- remaining limitations
- whether any roadmap/contract changes are required

---

# 42. IMAGE / ARCHITECTURE UNDERSTANDING

If repository documentation contains architecture diagrams, screenshots, charts, UI images or other visual artifacts:

**Do not ignore them.**

When available:
1. inspect the image;
2. connect the visual architecture to the written contracts;
3. verify that implementation matches both;
4. do not infer undocumented behavior from visual appearance alone.

For chart-related work, visual chart inspection can complement numerical feature analysis, but production ML features must remain formally defined and reproducible.

---

# 43. IMPORTANT DISTINCTION: SCORE VS FORECAST VS DECISION

Keep these layers separate:

```text
                    EVIDENCE
                       |
          +------------+------------+
          |                         |
          v                         v
  INVESTMENT SCORE             FORECAST ENGINE
          |                         |
          |                    1D/5D/20D/
          |                    60D/120D...
          |                         |
          +------------+------------+
                       |
                       v
                RISK + AUDITOR
                       |
                       v
                  DECISION
                       |
          +------------+------------+
          |                         |
          v                         v
       PAPER                    NO_TRADE
```

Do not allow one layer to silently perform the responsibilities of another.

---

# 44. THE END STATE

The mature system should eventually behave like this:

```text
User selects:
NVDA
as_of = current time

        |
        v

SYSTEM RECONSTRUCTS EXACT KNOWLEDGE STATE

        |
        +-- Price history
        +-- Chart state
        +-- Fundamentals
        +-- News
        +-- Events
        +-- Influential people
        +-- Sentiment
        +-- Macro
        +-- Market regime
        +-- Sector context
        +-- Historical analogs
        |
        v

INVESTMENT QUALITY SCORE

        +

MULTI-HORIZON FORECAST

        |
        +-- 1D
        +-- 5D
        +-- 20D
        +-- 60D
        +-- 120D
        |
        v

UNCERTAINTY + CALIBRATION

        |
        v

RISK

        |
        v

AUDITOR

        |
        v

FINAL POSTURE

PAPER / ANALYSIS_ONLY / NO_TRADE

        |
        v

LATER:

ACTUAL OUTCOME ARRIVES

        |
        v

FORECAST ERROR

        |
        v

PERFORMANCE / DRIFT / CALIBRATION

        |
        v

CANDIDATE MODEL UPDATE

        |
        v

VALIDATION

        |
        v

HUMAN-APPROVED PROMOTION

        |
        v

NEW CHAMPION
```

---

# 45. FINAL PRINCIPLE

The project's ambition is high, but the implementation philosophy must remain disciplined.

The objective is **not** to build the most complicated AI.

The objective is to build a system that can eventually make statements like:

> "Based on everything that was actually knowable at 14:32 UTC, this event occurred, this chart state existed, the market was in this regime, these historical situations were comparable, and the validated model estimated this future distribution with this calibrated uncertainty."

And then, later:

> "Here is what actually happened, how wrong the forecast was, why it was wrong, and whether the model should learn from it."

That is the standard.

**Evidence first.  
Point-in-time correctness first.  
Determinism first.  
Validation before promotion.  
Risk and audit can veto.  
ML earns its place through out-of-sample evidence.  
Historical forecasts never change.  
New information becomes new evidence.  
New evidence becomes new learning opportunities.**

This is the long-term architecture of `invest-by-score`.
