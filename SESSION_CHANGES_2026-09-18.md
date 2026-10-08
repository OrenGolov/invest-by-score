# Session changes — 2026-09-18

Branch `sprints-vs`, base commit `779ee6d`. **All changes are uncommitted.**

Verification at time of writing: **1168 tests pass** (from 1149 at base), **16/16 gate
scripts pass**, dashboard rendered and confirmed in a headless browser with 0 console
errors.

---

## 1. Environment

Created `.venv` (Python 3.13.5) and installed all 29 pinned dependencies from
`requirements.txt` — `scikit-learn` and `scipy` were missing from the global
interpreter, so the suite could not run at all before this.

Run everything through `./.venv/Scripts/python.exe`.

---

## 2. Portfolio additions: RCAT + ASTS

`RCAT` and `ASTS` had been added to `PORTFOLIO_TICKERS` in `fetch_data.py` as an
uncommitted local edit. That edit alone left the system inconsistent: **4 tests were
failing** and the point-in-time universe status had degraded from
`point_in_time_complete` to `unverifiable`.

Two registries track the universe and both were missing the tickers:

| File | Change |
|---|---|
| `data/universe.jsonl` | Seeded both via `seed_portfolio_universe` (74 → 76 entries). Written through the seeding function so the entry hashes match, keeping re-seeding idempotent. |
| `core/entity_resolution.py` | Registered `AST SpaceMobile Inc.` (Abel Avellan) and `Red Cat Holdings` (Jeff Thompson). |

Both tickers now fetch live data and score end to end.

### Bug found: bare ticker tokens matched ordinary words

Adding RCAT surfaced a pre-existing defect. `CAT` is three characters, so it passed the
short-ticker guard and matched the **word** "cat":

```
"The cat sat on the mat"          -> CAT (Caterpillar), confidence 1.0
"Red Cat Holdings wins contract"  -> ambiguous (CAT + RCAT) -> excluded from training
```

Ticker matching is now **case-sensitive**: `CAT` matches, `Cat`/`cat` do not. Names,
aliases and executives remain case-insensitive, because those are prose. The dead
`_tokens` helper it replaced was deleted.

7 regression tests added (`TestTickerCaseSensitivity`, `TestNewPortfolioMembers`).

---

## 3. Audit of commit 779ee6d

Both bug fixes in that commit are **real and correctly implemented**. The chain fix
classifies against the previous link while anchoring the window to the initial claim;
`negative → positive → negative` still records two reversals as claimed.

Two inaccuracies in the commit message, neither affecting its conclusions:

- **Test count.** Claims "10 new regression tests (4 chain, 6 similarity)". There are
  **11** (4 chain, 7 similarity).
- **The `0.654` before-figure does not reproduce.** Recomputing the pre-fix formula on
  the real 5-field fixture gives **0.5848**. The message quotes an abbreviated 3-field
  example next to 5-field numbers. The post-fix **0.968 is exactly right** (0.967906),
  and both before-values sit below the 0.7 retrieval bar, so the finding stands.

One substantive weakness fixed: `test_the_window_is_still_measured_from_the_initial_claim`
asserted only `len(chains) > 1`, which passes for 2 *or* 3 chains and so did not pin the
boundary the commit says it pins. Tightened to assert exactly 2 chains with link counts
`[2, 1]`.

---

## 4. Flaky test fixed: `RawStoreRebuildTests`

This intermittently failed with `AssertionError: 92.5 != 94.5`.

**Root cause.** `data_quality.score` is computed as
`quality_score -= future_bars_excluded * 0.5`. The test calls `fetch_market_snapshot`
twice, and the second call honours the default 24h cache TTL — once it expires the call
re-fetches from Yahoo and sees a different bar count. `92.5` was 15 stale future bars;
`94.5` is 11.

Proven rather than assumed: aging the cache to 72h and running the **original** test
reproduced the exact original error; the fixed version passes under identical conditions,
and the full suite passes with all 13 parquet caches expired.

The fix neutralises the TTL check itself. A first attempt patched
`fetch_data.DEFAULT_CACHE_TTL` and silently did nothing — that constant is bound as a
default argument at definition time. The working version patches `_read_cache`.

---

## 5. Sprint N (N3): VIX + 30Y yield added to the macro registry

10Y (`DGS10`) was already registered. Added two series, both following the existing 10Y
pattern exactly and inheriting N3's publication-time PIT gating and first-release
retention:

| Key | FRED id | Frequency | Role |
|---|---|---|---|
| `30y_yield` | `DGS30` | daily | ultra-long rate / term premium |
| `vix` | `VIXCLS` | daily | implied volatility — **inverted** |

Registry is now **7 series**.

### VIX is inverted — the main risk in this change

Every other macro signal reads "higher = more risk-on". A high VIX is the market pricing
fear, so its sign is flipped. A sign error here would turn a panic into a buy signal.

Pinned three ways: a test asserting VIX 38 → `risk_off` while 30Y 4.8 → `risk_on` in the
same test; the series description stating the inversion; and a test asserting the
description says so.

Bands (`core/config.py`): VIX 30 stress / 20 elevated / 14 calm. 30Y 4% / 3% / 2%,
one notch above the 10Y to reflect the term premium.

### Widening the signal set changes historical scores

`risk_score` is the **mean** of available signals, so each signal's share moved from 1/5
to 1/7. The `HEALTHY` fixture moved `0.58 → 0.6071` (macro_score `7.9 → 8.0355`), still
`neutral`. This is why `MACRO_ADAPTER_VERSION` moved **v2 → v3** rather than the series
being added silently. A test asserts the dilution is intentional.

Each changed assertion was recomputed rather than loosened, and several hardcoded `5`s
were replaced with `len(MACRO_SERIES_REGISTRY)` so the next series addition will not
break them.

### Incidental cleanup

`MACRO_RISKOFF_THRESHOLD` and `MACRO_RISKON_THRESHOLD` were each **defined twice** in
`core/config.py`, comment included. Same values, so harmless — but a trap if someone
edited one copy. Collapsed to one definition.

---

## 6. Dashboard: scoring weights panel

### Server-side gap

`orchestrate_score` **computed `ensemble_breakdown`, used it internally for the audit,
then discarded it** before returning the decision. The dashboard had no way to show which
agent carried a score — only the number.

- `core/agent_contracts.py` — `OrchestrationDecision` gains an optional
  `ensemble_breakdown` field (defaults to `{}`, so nothing existing breaks).
- `core/orchestrator.py` — populates it from `score_result`.

### New panel

A **Scoring Weights** panel in `index.html`, rendered below the score for every searched
ticker:

1. **Macro context — pinned to the top, in this order: VIX, 10Y, 30Y.** These are macro
   *series* (inputs to the macroeconomic agent), not ensemble agents, so they carry no
   weight of their own and render as context with an explicit status rather than a number
   that would look like a weight. Read from the macro agent's `series_values`.
2. **Ensemble weights (current-time)** — heaviest first, each with agent status and the
   renormalized effective weight where it differs from the raw weight.
3. **Ensemble weights (long-term)** — same, using the long-horizon effective weights.
4. Ensemble version.

### Verified in a real browser, not assumed

Rendering caught two bugs that code review would not have:

- The panel initially read `data.*`, but `render()` receives the API **envelope** and
  unwraps to `decision`. It showed "No ensemble weights available" until fixed.
- Both weight sections used `effective_weight_current`, so the long-term section showed
  12.5% for a 20% raw weight. Now uses `effective_weight_long` — correctly 22.2%.

Confirmed with populated values injected through the real render path:
**VIX 17.4, 10Y 4.21, 30Y 4.55** at the top, weights below, 0 console errors.

4 contract tests added (`DashboardWeightsContractTests`).

---

## Files changed (15, all uncommitted)

```
INVEST_BY_SCORE_MASTER_CONTEXT.md   §17: registered-series table + VIX inversion rule
core/agent_contracts.py             ensemble_breakdown on OrchestrationDecision
core/config.py                      VIX/30Y bands, adapter v3, duplicate constants removed
core/entity_resolution.py           case-sensitive tickers, RCAT + ASTS registered
core/macro_adapter.py               VIX + 30Y signal blocks
core/macro_registry.py              VIX + 30Y series entries
core/orchestrator.py                carry ensemble_breakdown into the decision
data/universe.jsonl                 RCAT + ASTS ledger entries
docs/task-breakdown.md              N3 updated: five -> seven series
fetch_data.py                       RCAT + ASTS (your edit, now consistent)
index.html                          Scoring Weights panel
tests/test_entity_resolution.py     +7 tests
tests/test_event_chains.py          tightened window assertion
tests/test_macro_registry.py        +8 tests, fixtures extended to 7 series
tests/test_scoring.py               +4 tests, flake fixed
```

`NEXT_STEPS_SPRINTS.md` was **not** changed — its Sprint N bullet describes the macro
registry generically and is still accurate.

---

## Open items

- **No API keys are set** (`ALPHAVANTAGE_API_KEY`, `NEWS_PROVIDER_API_KEY`,
  `FRED_API_KEY`). Macro reads `UNAVAILABLE`, so VIX/10Y/30Y contribute nothing and show
  as `UNAVAILABLE` in the dashboard until `FRED_API_KEY` is configured. Verified still
  fail-closed.
- **Scores barely differentiate without a fundamentals key.** RCAT and ASTS both score
  1.04, because the only directionally-weighted agent that is OK is
  `fundamental_analysis`, which returns a flat 6.0 for every ticker with no key. The
  technical agent carries weight 0.0 ("informational only"). Worth setting
  `ALPHAVANTAGE_API_KEY` before Sprint C, which is chart-focused.
- **`get_provider_health_matrix()` reports nominal registry state, not resolved runtime
  state.** It shows `fundamentals: live_provider, health=0.9` while the actual resolver
  returns `provider_key_required, confidence 0.0`. Its test only asserts keys exist.
  Pre-existing; it feeds a web endpoint.
- **Alerting does not exist** — no code anywhere, and Sprint A is absent from
  `docs/task-breakdown.md`. Deferred by agreement.
- Untracked and left alone: `PY` (a stray VS Code artifact) and today's
  `data/raw/*.jsonl` fetch logs. The old local `sprint-vs` branch is untouched.
