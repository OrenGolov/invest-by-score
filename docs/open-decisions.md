# Open decisions and deferred items

The running register of things deliberately **not** done yet. A decision lands
here when it is a judgement to make rather than a defect to fix, or when it is
real work parked until a later sprint.

**This file is not a bug list.** A defect gets fixed or it blocks a sprint; it
does not get registered and forgotten. Everything here is either awaiting an
operator decision or awaiting its sprint.

Each entry carries the **measurement or reason** that justifies parking it, so
the decision can be re-made later on evidence rather than on memory. Items are
closed by striking them through with the commit that resolved them, never by
deletion — a register that forgets what was decided is a register that gets
re-litigated.

Status: `DECISION` (needs the operator) · `PARKED` (needs its sprint) ·
`CLOSED`.

---

## 1. VOO news tracking — DECISION

**Raised:** Sprint L (2026-09). **Owner:** operator.

VOO is skipped for news collection as a broad-market fund whose news E5 would
classify as confounded: an S&P 500 move has no single attributable cause, so an
event study against SPY as its own benchmark measures nothing.

Against that: the S&P 500 is one of the most important series the system
follows, and the operator has asked to revisit rather than accept the default.

**What would settle it:** whether a broad-index event study can produce a
non-confounded reading against a *different* benchmark (equal-weight, or a
macro factor), or whether VOO news is better consumed as regime context than as
an event. Not a defect — a judgement to re-make deliberately.

## 2. NASA has no metadata — DECISION

**Raised:** Sprint L (2026-09). **Owner:** operator.

`data/universe.jsonl` carries NASA's ticker and nothing else, so it cannot be
classified as a fund or a company and is skipped for news.

MEASURED 2026-09-21: it also holds only **119 bars**, below the 210 a chart
state requires, so it is refused by live forecasting for an unrelated second
reason. Both resolve the same way — decide what the instrument is and whether
it belongs in the universe at this history length.

## 3. Breadth / participation features — PARKED

**Raised:** Sprint C. **Blocked on:** an index-constituent adapter.

Explicitly deferred with a `breadth` entry in `fetch_data.SOURCE_REGISTRY`
(`provider_key_required`, `base_confidence: 0.0`, `domain: market_breadth`).

**Why it cannot simply be added:** survivorship-safe breadth requires knowing
which names were *in* the index on each past date. It can never be inferred
from price, volume or technical indicators — inferring it from today's
constituents is survivorship bias by construction. Held open by 18 tests in
`tests/test_deferred_features.py`, including the placeholder itself.

## 4. The source → outcome join does not exist — PARKED

**Raised:** Sprint L4 (2026-09). **Blocks:** everything L4 can actually learn.

L4 ships a working source-reliability estimator, but it has nothing to learn
from, because no outcome in the system is attributable to an OUTLET.

MEASURED 2026-09-21:

```
2650 stored articles carry 50 distinct `source_name` values
   0 of them carry a ticker, so no article joins to a price outcome
2084 event memories carry NO outlet at all
     (1954 are inferred from volume cadence and have no source by construction)
Event.source is set to the PROVIDER ("newsapi_news"), not the outlet
```

**Why this is parked rather than fixed inside L4.** Building the join means
attributing a price outcome to a specific article — which is an event-study
question (E4/E5), not a scoring question. Doing it here would duplicate the
attribution machinery those sprints own (W5).

**What it would take:** articles resolved to tickers at ingestion (N-sprint
entity resolution already does this for events), `Event.source` carrying the
outlet alongside the provider, and the outlet surviving into `EventMemory`.
Until then every L4 score is a registry prior and reports itself as such —
the estimator degrades correctly, it is simply not yet learning.

**Scale note:** the full conditional scheme is 19,800 cells needing ~7.5M
source-linked outcomes to populate. Even with the join built, the useful
near-term output is per-source global rates, not the full cross-product.

## 5. Cluster exposure is not measured — PARKED

**Raised:** Sprint R1 (2026-09). **Blocked on:** a later R task.

R1 flags exposure per POSITION, on both weight and risk. A CLUSTER of
correlated holdings is a different question and R1 cannot see it.

MEASURED on real returns, a portfolio of 10% each in NVDA/AMD/AVGO/SOXX plus
MSFT/GOOGL/CAT:

```
no single holding exceeds the 40% risk threshold
  CAT 17.9%   AMD 17.5%   GOOGL 14.4%   AVGO 13.9%

but the four semiconductors TOGETHER hold 56.9% of portfolio variance,
at a mean pairwise correlation of 0.62
```

That portfolio raises **zero flags** while being more volatile (1.760%) than
one holding 40% NVDA outright (1.741%).

**Why it is parked rather than bolted onto R1.** Defining a cluster needs its
own evidence — by sector, by correlation clustering, or by factor loading —
and each choice is a measurement, not a preference. Widening R1's scope to
guess at one would be the kind of unmeasured decision this project avoids.

## 6. Verify against a CLEAN CLONE, not the working directory — CLOSED-AS-PROCESS

**Raised:** 2026-09-21, after CI failed on nine consecutive pushes while every
local run reported green.

**Cause.** `data/event_memory.jsonl` is gitignored (2084 rows locally, absent
on a fresh clone). One gate and one test asserted that `record_run` always
reports `chart_refusals` — but `record_run` returns early when the memory
store is empty, correctly refusing to forecast from nothing, and that early
return reaches no chart stage. Both passed on my machine and failed on every
push.

**The deeper problem was my verification, not the code.** I ran
`python -m unittest` while CI runs `pytest`, and I ran in a working directory
carrying untracked data CI never sees. Neither difference was visible from
inside the loop I was using.

**Standing practice from here:**
1. Verify with `python -m pytest tests/ -q`, the runner CI actually uses.
2. Before reporting a sprint task green, run the gates and suite against a
   fresh `git clone` of the branch, not the working tree.
3. A gate or test that needs gitignored data must either tolerate its absence
   or generate what it needs.

## 7. Collection cadence is manual — DECISION

**Raised:** Sprint L (2026-09). **Owner:** operator.

The daily collector and the forecast runner are both correct to schedule now
(the live chart-state blocker closed in `ad75f48`), but nothing is scheduled:
the operator asked to run "every other day if needed, but not now".

**What would settle it:** whether to cron them, and at what cadence. The
NewsAPI free tier's 100 requests/day is the binding constraint, which is why
`COLLECT_NEWS_TRACK_ANYWAY` exists for SOXX and CIBR.

---

## 8. `test_current_and_long_term_scores_diverge_by_regime` is network-flaky — OPEN

**Raised:** Sprint A (2026-09-22). **Owner:** unassigned.

MEASURED on a fresh clone at `415ea7b`: the full suite failed once on
`tests/test_scoring.py::ScoringEngineTests::test_current_and_long_term_scores_diverge_by_regime`,
then passed on an immediate rerun (2684 passed). The test passes in isolation.

**Cause.** The test calls `build_score(...)` for five tickers at `2024-01-02`,
which reaches `fetch_price_history` -> Yahoo Finance through an on-disk cache.
When a fetch degrades, the five scores collapse toward each other and the
test's `len(set(...)) > 1` assertion fails. It is a live-network dependency in
a unit test, not a scoring regression: A1 added four files and touches no
scoring code.

**Why it matters.** A flaky test trains the reader to ignore a red suite,
which is exactly how the nine-run CI failure in `9e7852d` went unnoticed. It
also means CI can go red for reasons unrelated to the commit under test.

**What would settle it:** pin the test to a recorded fixture (the raw ledger
already stores digest-addressable payloads, so a replay is available), or mark
it as requiring network and exclude it from the default run. The first is
preferable — it makes the test deterministic rather than merely quieter.

---

## Closed

- ~~**Live runs must use today's chart state.**~~ Fixed in `ad75f48`. The
  memory-derived state was a median 144 days stale and retrieved analog sets
  overlapping the live ones by a Jaccard of 0.205, calling VOO and CIBR
  bearish while both were bullish.
