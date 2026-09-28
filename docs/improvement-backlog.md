# Improvement Backlog — invest-by-score

> **Status:** DRAFT for review. Nothing here is started.
>
> Compiled 2026-09-28, after Sprint X (X1–X10) closed. Every item below is
> backed by a measurement taken against the repository at `7add1dd`, not by
> impression. Where a number appears, the command that produced it is named.
>
> **The organising fact:** the platform's *machinery* is in good shape — 3,417
> tests, 71 gates, 90 core modules, W5 holding, fail-closed behaviour verified
> under a real provider outage. What it lacks is **evidence**: X10 reports
> `FORECASTING RELEASE: NOT APPROVED` with **1 of 9 gates passing**, and six of
> the nine cannot run at all. So this backlog is weighted toward *making the
> gates runnable*, not toward more gates.

---

## Where the project actually stands

Measured at `7add1dd`:

| | |
|---|---|
| core modules / lines | 90 / 49,324 |
| `core/config.py` | **9,584 lines (20% of core)**, 1,077 constants, 73 validators |
| tests / files | 3,417 passing, 1 skipped / 88 files, 34,179 lines |
| governance gates | 71, all green on a clean clone |
| gate wall-clock | **126 s**, of which `check_reproducibility.py` is **106 s (84%)** |
| suite wall-clock | 175 s, of which two test *setups* are 70 s |
| release verdict (X10) | **NOT APPROVED — 1 of 9 gates pass, 6 unevaluable** |
| dependencies | fully pinned (28 packages) |
| CI | pytest + all gates discovered; **no linter, no type checker** |
| open decision items | 11 (3 OPEN, 3 PARKED, 3 DECISION, 2 CLOSED) |

**What is genuinely good and should not be disturbed:** the fail-closed
contract holds under real failure (a live NewsAPI 429 produced `UNAVAILABLE`,
not a fabricated neutral); dependencies are pinned; secrets are read from env
vars only, never hardcoded; gates are *discovered* by CI rather than
enumerated; W5 holds (one canonical implementation — no split-brain duplicates
found).

---

## The five themes

Ordered by how much each unblocks:

1. **A → Make the evidence exist.** Six gates cannot run; five share *one*
   defect (the thin fold record). And separately: 24 of 40 registered features
   never reach a model. This is the only theme that changes the release verdict.
2. **B → Close the wiring gaps.** Things that are built, tested, and connected
   to nothing.
3. **C → Pay down structural debt.** `config.py` at 9,584 lines; 106 s in one
   gate.
4. **D → Raise the engineering floor.** No linter, no type checker, no coverage
   measurement.
5. **E → Resolve the standing decisions.** The 11 open items, several of which
   are one-line calls.

---

# Sprint A — Make the evidence exist

**Goal:** move the X10 verdict off `1 of 9`. This is the sprint that matters;
everything else is housekeeping beside it.

**Why it is one sprint:** X10 measured that five of the six blockers share a
single prerequisite. They are not five pieces of work.

### A1 — Enrich the training fold record *(unblocks X3, X4, X5, X8)*

**Measured:** the fold carries exactly eight keys — `actuals`, `fold_id`,
`metrics`, `predictions`, `train_end_time`, `train_rows`, `validation_rows`,
`validation_start_time`. It carries none of `regimes`, `events`, `sources`,
`validation` (absolute indices), `ablation`.

Add per-observation context to `FoldResult`, written by `train_baseline`:
- `regimes` — the regime label at each prediction time → **unblocks X3**
- `events` / `sources` — event id and source at each prediction time → **X4**
- `validation` / `train` — **absolute row indices**, not counts. `train_rows`
  says *how many*, never *where*, and only position can prove a fold stopped
  short of the holdout → **X8**

**Acceptance:** each of X3, X4, X8 moves off `NOT_EVALUATED` on real runs —
to `ROBUST` *or* to a genuine failure. Either is a result; `NOT_EVALUATED` is
not. The existing gates already fail if these keys appear without the gate
being re-run, so they will tell us when this lands.

**Risk to watch:** do not synthesise a label to fill a column. X4's module
already refuses to derive attribution for exactly this reason — bucketing every
observation under one synthetic id leaves leave-one-out nothing to leave, and
the gate would return `ROBUST` having tested nothing.

### A2 — Extend the dataset past the geometry floor *(unblocks X2, X8; open item 11)*

**Measured:** 406 rows available. The cheapest *legal* walk-forward geometry
(fold 60 / embargo 252 / holdout 60) needs **432** — short by 26. The current
default (252/252/126) needs **882** — short by 476.

Decide and execute one of:
1. **extend the dataset past 432 rows** (~500 for comfort), retrain, reseal —
   the only option that preserves what F2 set out to do;
2. **drop the 252-session horizon** if a one-year label is not actually wanted,
   returning the embargo floor to 120;
3. **keep both** and document which horizons a run may legitimately claim.

**Also fix regardless of the choice:** `scripts/train.py` ships
`--embargo-sessions 60` as a default, which `build_walk_forward_folds` now
rejects outright. **The script cannot run with its own defaults.**

**Acceptance:** `scripts/train.py` regenerates `data/training_runs.jsonl` from
its defaults; X8 reaches `SEALED`; X2 scores calibration on a real holdout.

### A3 — Train across horizons *(unblocks X6)*

**Measured:** all 8 runs train the single `20d` target, so X6 has no horizon
spread to compare and reports `HORIZONS_MISSING`.

Train the declared set (1d/5d/20d/60d/120d/252d), subject to A2 — the 252d
horizon may not be trainable on the available history, which is itself the
honest finding to record.

### A4 — Get non-price features into the training pipeline *(unblocks X5; the biggest gap on this list)*

**Measured:** the training dataset produces **16 features, and all 16 are price
or volume derivatives**:

    change_1d change_5d change_20d change_60d
    ma_50 ma_100 ma_150 ma_200
    price_vs_ma_50 price_vs_ma_100 price_vs_ma_150 price_vs_ma_200
    rsi trend_vs_20d_mean volatility volume_ratio_20d

X5 put it exactly right: **four of the five named sources were never there to
remove.**

| group | features in the pipeline |
|---|---|
| technicals | 16 |
| news | **0** |
| macro | **0** |
| sentiment | **0** |
| fundamentals | **0** |

**This is the most consequential finding in this document, and it is larger than
X5.** The project's thesis is that news, events, macro, sentiment and
fundamentals carry predictive information. Sprints N, E, C and F built adapters,
registries, event memory and forecasting on top of all five. **None of them reach
a model.** Every trained estimator in the ledger is a technical model.

So the honest reading of the whole ML layer is narrower than the roadmap implies:
we have not yet tested the project's central hypothesis, because the features
that would test it have never entered a training row.

**The gap is narrower than "the features don't exist", which makes it
tractable.** They are already REGISTERED in M1 — they are simply never PRODUCED.
Measured: **40 features registered, 16 used in training, 24 never trained on**,
and the 24 are exactly the interesting ones:

| family | registered but never trained on |
|---|---|
| fundamentals | `revenue_growth`, `valuation_quality`, `margin_quality`, `free_cash_flow_quality`, `balance_sheet_quality` |
| news / sentiment | `news_sentiment_score`, `sentiment_score` |
| macro / regime | `macro_regime_score`, `regime_probability_proxy`, `volatility_regime_ratio` |
| chart (C2) | `atr_14`, `trend_slope_60d`, `gap_pct`, `drawdown_60d`, `recovery_speed_60d`, `breakout_state_60d`, `support_distance_60d`, `resistance_distance_60d`, `relative_strength_60d`, `acceleration_10d` |
| longer returns | `change_50d`, `change_100d`, `change_150d`, `change_200d` |

So M1 (the registry) and the producers exist on paper; the **M2 dataset builder
emits only the 16 price/volume columns**. The work is connecting registered
features to the dataset builder, not inventing them.

**Acceptance:** at least one non-price family is present in a trained run, and
X5 reports whether it adds incremental value. **A negative answer is a real
result** — and arguably the most valuable single measurement the project could
make, because it would tell us whether the whole N/E/C evidence layer earns its
place.

**Sequencing note:** the 10 chart features (C2) and 4 longer returns are pure
price derivatives with producers already written — they are the cheapest to wire
and would let X5 run at all. The fundamentals/news/macro families are the ones
that test the actual thesis. Do the cheap ones first to prove the wiring, then
the meaningful ones.

**Dependency:** needs B2 (the source → outcome join) for the news/source
families specifically.

### A5 — Register every trial *(improves X7)*

**Measured:** `data/research_trials.jsonl` holds **2 rows that are 1 distinct
trial**, against **8 trained estimators**. X7 correctly corrects on the observed
family of 8 and reports the discrepancy, but the registry is the mechanism M3
built to prevent exactly this.

Route every training run through trial registration. **Acceptance:** the
registry count equals the estimator count, and X7's registry-gap finding goes
quiet on its own.

---

# Sprint B — Close the wiring gaps

**Goal:** the W-sprint principle is *"every agent is born wired."* These are
components that exist, pass tests, and are reachable from nothing.

### B1 — `paper_engine` is not wired *(HIGH — a roadmap claim that is not true)*

**Measured:** `core/paper_engine.py` is 259 lines with a tested
`submit_order_intent`. It is imported by **`tests/test_paper_engine.py` and
nothing else**. Meanwhile `core/score_engine.py:472` emits the mode
`"PAPER_TRADING_READY"`.

So the system announces paper-trading readiness while no code path can place a
paper order. Either wire it behind the governance gate, or rename the mode to
what is true. **My recommendation: wire it** — V's paper engine is a real
roadmap deliverable, and the governance plumbing to gate it already exists.

**Acceptance:** a decision that passes risk + auditor produces a paper order
intent; one that fails produces none, with the refusal recorded.

### B2 — The source → outcome join does not exist *(open item 4, PARKED)*

Blocks L4 (source reliability learning) from measuring anything, and is a
prerequisite for the news/source half of A4. Worth unparking precisely because
A4 now depends on it.

### B3 — F7's structured assessment has no consumer contract *(open item 10)*

Two independent consumers hit the same ambiguous shape in one sprint. A third
is coming. Export a documented accessor or add a named scalar field.

### B4 — Cluster exposure is not measured *(open item 5, PARKED)*

**Measured in R1:** a 4×10% semiconductor portfolio raises **zero flags** while
holding **56.9% of portfolio variance** in one bet at mean pairwise correlation
0.62. R3 addressed sector concentration; correlated *clusters* that cross sector
labels are still unmeasured.

---

# Sprint C — Structural debt

**Goal:** make the codebase navigable and CI affordable. No behaviour changes.

### C1 — Split `core/config.py` *(9,584 lines, 20% of all core code)*

**Measured:** 1,077 constants, 73 import-time validators, 65 `# ---` sections,
38% comment lines. Every sprint appends a block; nothing is ever removed.

The comments are *load-bearing* — they carry the measurement that justifies each
rule, which is the project's best documentation. **Do not delete them.** Split by
sprint into `core/config/` (`__init__.py` re-exporting everything so no import
changes), e.g. `_governance.py`, `_forecasting.py`, `_learning.py`,
`_portfolio.py`, `_release.py`.

**Acceptance:** `from core.config import X` works unchanged for all 1,077
constants; all 73 validators still run at import; suite and gates unchanged.
This is a pure refactor and should be verified by *zero* test changes.

**Prerequisite:** do this only with the full suite green and against a clean
clone — it touches every module.

### C2 — `check_reproducibility.py` takes 106 s of the 126 s gate budget

**Measured:** 105,803 ms. Every push pays it. Options: cache the trained
artifacts it compares, reduce the estimator set to a representative sample
(justified by measurement, not convenience), or split it into a fast
subset-check on every push plus a nightly full run.

**Guard against the obvious mistake:** do not weaken *what* it proves to make it
faster. If a sampled version cannot detect a reproducibility break, it is not a
cheaper gate — it is a broken one. Sabotage-test any replacement.

### C3 — Two test setups cost 70 s *(LOWER PRIORITY than it first looks)*

`test_backtest.py::test_cost_parameters_demonstrably_change_results` setup is
**44.5 s**; `test_reproducibility.py::test_every_baseline_is_reproducible` setup
is **26.0 s**.

**A claim I checked and withdrew:** my first reading was that these are per-test
fixtures that should be class-scoped. They already ARE — both use `setUpClass`.
So the 70 s is genuine work (training estimators, running walk-forward
backtests), not a scoping bug, and there is no cheap win here.

The only real lever is sharing the trained artifacts between the backtest and
reproducibility suites, which couples two independent test modules — likely not
worth it for 70 s on a 175 s suite. **Recommend: leave alone unless the suite
grows past ~5 minutes.** Listed for completeness, not as work.

### C4 — `snapshot_problems` is defined in three modules

`forecast_snapshot` (F8), `release_snapshot` (X9), `timeframes` (C1). These are
three genuinely different concepts, so this is **not** a W5 violation — but the
shared name will mislead. Rename to `forecast_snapshot_problems`,
`release_snapshot_problems`, `timeframe_problems`.

---

# Sprint D — Raise the engineering floor

**Goal:** catch mechanically what review currently catches by hand.

### D1 — Add a linter to CI *(there is none)*

**Measured:** zero references to ruff/flake8/mypy/black/lint in
`.github/workflows/`, and no lint config anywhere in the repo.

Add `ruff` with a config matching existing style rather than imposing a new one.
Expect a large first-run diff on 49k lines; land the config and a
`# noqa`-annotated baseline separately from any fixes.

### D2 — Add a type checker to CI

The codebase already annotates heavily (`from __future__ import annotations`
throughout, typed signatures). `mypy` or `pyright` in non-strict mode would pay
for itself. Start with `core/` only.

### D3 — Measure test coverage

**Measured:** 89 of 90 core modules are *imported* by some test — but import is
not coverage. No coverage tooling is configured, so which *branches* are
exercised is unknown. Add `pytest-cov`, publish the number, and set a floor
*below* the current value so it ratchets rather than blocks.

### D4 — Audit the 19 `except Exception` sites in core

**Measured:** 19 sites, concentrated in `score_engine.py` (4) and
`release_snapshot.py` (4), with 11 more spread across `training_dataset`,
`timeframes`, `sequences`, `regime_agent`, `promotion`, `portfolio_decision`,
`market_context`, `labels`, `forecast_snapshot` and `backtest/manifest`.

Some are deliberate and correct — X9's `_probe` converts a probe failure into an
explicit `ABSENT`, which *is* the designed behaviour, and `_git` returns `None`
rather than a sentinel string. Others may be swallowing real errors in a system
whose whole contract is that failures must be loud. Review each; where the catch
is intentional, say so in a comment naming what it absorbs and why.

### D5 — A gate that catches checkout-dependent assertions

**Measured: this bug class occurred three times in two days** — `9e7852d` (gate
needed gitignored data), `2aa15f2` (same bug inverted), `755e94b` (assertion
keyed to whether the tree was clean). Open item 6 now records the general rule.

A meta-gate could catch it mechanically: run the gate suite twice — once with
gitignored data present, once with it moved aside — and fail on any gate whose
verdict differs. That is a real, cheap check for a defect that has already cost
three fixes.

---

# Sprint E — Resolve the standing decisions

Mostly one-line calls that are cheap to make and cheap to get wrong by drift.

### E1 — Collector cadence *(open item 7 — ACTIVE CONSEQUENCE)*

**Measured:** the collector has not run since **2026-09-21**. Sept 22–28 are
missing (5 business days, allowance 2), so `check_data_coverage.py` fails on
this machine. It correctly *passes* on a fresh clone, because it refuses to
judge a machine that has never run the collector.

Untracked output from a run today is sitting in `data/raw/`. Two decisions:
whether to commit it, and whether to register the scheduled task
(`scripts/install_daily_task.ps1`). **The NewsAPI free tier is the binding
constraint — a live call today returned HTTP 429.**

### E2 — NewsAPI quota is exhausted *(NEW — flagged proactively)*

A live `build_score` call returned `HTTP Error 429: Too Many Requests`. The
system degraded correctly (news `UNAVAILABLE`, not a fabricated neutral), so
this is not a bug — but it means **news evidence is currently unavailable to
every forecast**, and any run made today carries that gap. Decide: accept the
free tier and batch harder, or upgrade.

### E3 — Unpark or close items 3, 4, 5

- **3. Breadth / participation features** — deferred until properly wired.
- **4. Source → outcome join** — now blocks A4/B2, so effectively promoted.
- **5. Cluster exposure** — see B4.

Each is "parked" rather than decided. Parking is fine; parking indefinitely
without a revisit date is how a register becomes a graveyard.

### E4 — Items 1, 2, 9 — confirm or close

**1.** VOO news tracking, **2.** NASA metadata, **9.** sabotage probes need a
plausible wrong answer available. All three are recorded decisions; confirm they
are still the intent and mark them closed, or reopen.

---

## Suggested order, and why

```
A  (evidence)        ─── the only theme that changes the release verdict
│
├─ A2 first: the geometry floor blocks A3 and half of A8
│
B  (wiring)          ─── B1 is a false claim in the output; fix early
│
C1 (config split)    ─── do while the suite is green, before more sprints append
C2 (106 s gate)      ─── pays back on every push
│
D  (floor)           ─── D5 is cheap and prevents a bug that recurred 3×
│
E  (decisions)       ─── E1/E2 are time-sensitive (data is being lost now)
```

**If only one thing gets done: A1 + A2.** Together they move four gates off
`NOT_EVALUATED` and make `scripts/train.py` runnable again. Everything else is
improvement; that is the difference between a platform that reports "no evidence"
and one that reports a result.

**But the single most important measurement on this list is A4.** Every trained
model in the ledger uses 16 price/volume features and nothing else, while **24 of
the 40 registered features — every fundamental, news, sentiment and macro one —
have never entered a training row.** The project's central claim, that this
evidence predicts returns, has never been tested. A1/A2 make the gates
*runnable*; A4 is what makes the answer *interesting*, in either direction.

**If only one hour: E1 and E2.** Data not collected today cannot be collected
retroactively — the news window closes. That is the one item on this list where
delay destroys something irreplaceable.

---

## What I deliberately did *not* put on this list

- **More gates or more sprints.** The system has 71 gates and cannot run 6 of the
  9 that matter. Adding a 72nd would be motion, not progress.
- **Rewriting anything that works.** The scoring engine, the fail-closed
  contract, the provenance ledger and the gate-discovery CI are all sound and
  measured. Leave them alone.
- **"Improving" the config comments.** They are 38% of the file and they are the
  reason a reader can tell why a threshold is 1.4 + 0.8·ln(cells). They are an
  asset.
