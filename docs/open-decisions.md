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

## 4. Collection cadence is manual — DECISION

**Raised:** Sprint L (2026-09). **Owner:** operator.

The daily collector and the forecast runner are both correct to schedule now
(the live chart-state blocker closed in `ad75f48`), but nothing is scheduled:
the operator asked to run "every other day if needed, but not now".

**What would settle it:** whether to cron them, and at what cadence. The
NewsAPI free tier's 100 requests/day is the binding constraint, which is why
`COLLECT_NEWS_TRACK_ANYWAY` exists for SOXX and CIBR.

---

## Closed

- ~~**Live runs must use today's chart state.**~~ Fixed in `ad75f48`. The
  memory-derived state was a median 144 days stale and retrieved analog sets
  overlapping the live ones by a Jaccard of 0.205, calling VOO and CIBR
  bearish while both were bullish.
