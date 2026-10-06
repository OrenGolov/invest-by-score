# News collection: root cause, fixes, and architecture

**Date:** 2026-10-06
**Scope:** the news collection pipeline, its quota handling, the alerts it
produces, and the ML signal that depends on it.

Every number below was measured on this machine against the data on disk. Where
something is an estimate or a judgement rather than a measurement, it says so.

---

## 1. Executive summary

News capture stopped on 2026-10-04 and produced **0% coverage** of the
75-ticker eligible universe for three days. The visible symptom was a daily
digest of 77 alerts whose explanation was wrong in every clause. The cause was
not the laptop being off: the Windows task ran on schedule
(`LastRunTime 10/06/2026 09:18:26`, `LastTaskResult 0`, `NumberOfMissedRuns 0`).

**Four defects, not one.** They compounded, which is why the system got stuck
rather than degrading:

| # | Defect | Effect |
|---|--------|--------|
| 1 | The events stage refetched every ticker the news stage had already fetched | A batch of 40 cost **80 requests** against a 50/12h allowance — the run could never complete |
| 2 | The rotation cursor was deliberately frozen on a 429 | Every run restarted at the same ticker, failed identically, and never reached the rest of the list |
| 3 | Quota was detected by searching the error text for `"429"` | NewsAPI's JSON-body rate limit carries no `"429"`, so it was misreported as a provider outage |
| 4 | Alerts described a rotation that had not happened | 77 identical messages, one of which described the single ticker that *was* tried as the one that was not |

**The deadlock.** Defect 1 guaranteed a mid-run quota failure; defect 2
guaranteed the next run would fail in the same place. The cursor sat at
position 5 from 2026-10-04 to 2026-10-06 and 49 of 75 tickers were never
reached at all.

**ML impact is real but bounded, and it is silent rather than corrupting.**
Only 2 of 40 registered features depend on news, both `null_policy=exclude`, so
a missing provider removes features rather than substituting fake neutral
values. The substantive loss is in event memory: **26 of 75 tickers (34.7%)**
have any observed memory and only **18 (24.0%)** reach the 5-analog minimum a
conditional forecast requires.

**Status after this change:** cost per run halved (25 requests, 50% headroom),
the rotation makes guaranteed forward progress, all five failure modes are
distinguished with actionable text, and coverage metrics are printed on every
run. A full sweep now completes in 3.0 days, inside the 7-day window after
which news is unrecoverable.

---

## 2. Root cause analysis

### 2.1 The quota arithmetic was wrong by a factor of two

`core/news_adapter.py` has no cache of any kind — every `build_news_snapshot`
call is a live HTTP request. Both pipeline stages call it for the same tickers
on the same date:

- `collect_news` fetches the batch.
- `collect_events` → `build_event_memory.forward` fetched **the same batch
  again**.

The config comment asserted "40 covers all 73 in two runs and leaves 60 calls
spare." That assumed 1 call per ticker. The real cost was 80 — above the
100/24h ceiling and far above the 50/12h rolling window that is actually
enforced.

Measured in today's report, the two stages did not even agree on how far they
got: `news` reported 1 unavailable ticker, `events` reported 40. The news stage
stopped at the first 429; the events stage had **no 429 handling at all** and
called the provider 40 more times after the quota was already gone.

### 2.2 Freezing the cursor caused the starvation it was meant to prevent

The old rule was "never advance on a 429, so untried tickers are not skipped."
The reasoning is sound in isolation and wrong in practice: the next run began at
the same index, hit the same exhausted quota on the same ticker, and stopped.
Tickers deeper in the list were never attempted on any run.

`data/collect_cursor.json` read `{"cursor": 5, "updated": "2026-10-04..."}` on
2026-10-06 — two days without movement.

### 2.3 The quota signal was inferred from prose

`daily_collect` decided a day was a quota day with `"429" in str(reason)`. That
works for `urllib`'s `HTTPError` (`"HTTP Error 429: Too Many Requests"`) and
fails for NewsAPI's documented `rateLimited` response, which arrives as
**HTTP 200 with `{"status":"error","code":"rateLimited"}`** and a message
containing no `"429"`. That path was classified as a generic provider failure
and alerted as an outage — sending the operator to check a network that was
fine. The same substring would also misfire on a ticker named "429 Inc".

Note the repo had already been burned by this class of bug: `daily_collect.py`
carries a comment about a previous version blaming a missing API key when the
real cause was the quota. The fix then was to name the cause in the message;
the cause was still being *guessed* from a string.

### 2.4 The alert text described a mechanism that had not run

The digest said, 77 times:

> the news provider was unavailable (no news was captured for AAPL on
> 2026-10-06; the collector rotates 75 of the eligible tickers per run, so this
> ticker was not looked at today)

- "rotates 75 per run" — the batch was 40; 75 is the *eligible* count.
- "was not looked at" — said about AMZN, which was looked at and returned 429.
- "the news provider was unavailable" — the provider was up and the key valid.

All 77 read identically, so a ticker that was never scheduled (nothing wrong)
was indistinguishable from one whose fetch failed (something wrong).

**The digest generator is not in this repository, and not on this machine.**
I searched: every file type in the repo; `Desktop`, `Documents`, `Downloads`,
`source`, `repos` for its exact strings (`Unprioritised`, `Action: Review`,
`not trade instructions`); every branch tip and the full commit history via
`git log --all -S`. No hits. The scheduled task runs only `daily_collect.py`
and `monitor_collection.py`, and the latter sends the *collection-health*
verdict, not a 77-ticker digest.

So the sender cannot be wired from here. What I did instead was give it a
stable entry point rather than leave the fix stranded in a library:

    python scripts/monitor_collection.py --explain-tickers
    python scripts/monitor_collection.py --explain-tickers --json

The JSON form emits `{ticker: {kind, reason, action}}` on stdout with the human
banner on stderr, so it pipes into a parser cleanly. Whatever composes that
email needs one command and one JSON shape; until it is changed, this is also
how you get the truthful answer for any ticker yourself.

Run against today's real report, the 77 identical alerts collapse to three
grouped causes:

    not_scheduled   74 tickers  queued for an upcoming run; nothing failed
    not_tracked      2 tickers  NASA, VOO - no sector, by design
    quota_exceeded   1 ticker   AMZN - WAS fetched, quota spent, self-clearing

---

## 3. Code changes

### `core/news_adapter.py` — typed failure taxonomy

Five kinds (`quota_exceeded`, `authentication_failed`, `provider_unavailable`,
`provider_key_required`, `unknown_error`), each with operator guidance naming an
action. Classification happens once, where the HTTP status and response body are
both in scope:

- `classify_http_status` — 429 → quota, 401/403 → auth, 5xx/408 → unavailable.
- `classify_provider_error` — matches NewsAPI's machine-readable `code` field
  first, message substrings only as a fallback.
- `HTTPError` is now caught **before** `URLError` (its own base class) so the
  status code is read rather than discarded into a string.
- `failure_kind` rides on the snapshot. The no-key stub is left byte-identical,
  because `NoKeyContractTests` pins that dict exactly.

### `scripts/daily_collect.py` — budget, reuse, progress-preserving cursor

- `_NEWS_CACHE` + `news_snapshot_cached`: one provider call per
  `(ticker, as_of)` per run. Keyed on the date so a multi-date backfill cannot
  serve one date's articles for another — a point-in-time violation, not just a
  stale read.
- `COLLECT_NEWS_REQUEST_BUDGET` enforced at call time, so a future
  batch-arithmetic bug cannot overrun the window silently.
- `_advance_cursor(served=...)`: advances by work **completed**. Skips nothing
  (the unserved tickers are exactly where the next run starts) and guarantees
  forward progress. If nothing at all was served it nudges by 1, so one
  permanently-failing ticker cannot pin the rotation.
- Stops on `quota`/`auth`/`no_key` — the kinds where every further call is
  certain to fail. Other kinds are per-ticker noise and do not end the run.
- `_order_by_staleness`: orders the batch stalest-first, so if the quota dies
  mid-run the tickers that *were* served are those closest to permanent loss.
  Deliberately reorders only *within* the cursor's selection — reordering the
  selection itself would let a ticker the provider never has articles for
  monopolise every run.

### `scripts/build_event_memory.py` — stop refetching, stop on quota

`forward()` takes an optional `news_fetcher` (the collector passes its cache)
and breaks on quota/auth instead of calling the provider for every remaining
ticker.

### `core/collection_monitor.py` — accurate alerts

- Auth failures are checked **before** the quota branch and alert as CRITICAL.
  They present the same "no news" surface as a spent quota but never clear on
  their own, and conflating them tells the operator to wait out a window that
  will never reopen.
- `explain_ticker_news_gap` returns `{reason, action, kind}` per ticker, for the
  five real cases: not tracked by design / fetched-and-failed (by kind) /
  fetched with no articles / not scheduled this run / no run recorded.

### `core/news_coverage.py` (new) — the metrics that were missing

Coverage %, quota consumption, failed requests by kind, oldest missing news
date, event-memory freshness. Reads only from disk, so it costs no quota and
works while the provider is down — which is when it is needed. Printed by every
non-dry run.

### `core/config.py` — the invariant that would have caught this

The validator now asserts **cost**, not batch size:

```
a batch of 40 costs 80 requests (2x: news + events refetch), which exceeds
the 45-request per-run budget. Lower the batch, raise the budget, or enable
COLLECT_NEWS_REUSE_SNAPSHOTS
```

The old check compared the batch against the 100/day ceiling assuming 1 call per
ticker, so the broken config looked safe. Two stale comments that now describe
the opposite of the behaviour were corrected rather than left to mislead.

---

## 4. Verification

| Check | Result |
|-------|--------|
| Full suite, baseline | 3448 passed, 1799 subtests, 0 failures |
| Full suite, after changes | **3466 passed, 1799 subtests, 0 failures** (baseline 3448; +18 new tests) |
| `tests/test_daily_collect.py` | 49 passed |
| `tests/test_collection_monitor.py` | 35 passed (8 new, pinning the alert bug) |
| `tests/test_news_coverage.py` (new) | 15 passed — the metrics module, including that an attempted-and-failed fetch is never counted as coverage |
| Governance gates | 68 of 69 pass. `check_data_coverage` fails **at baseline too** — pre-existing, and it is correctly detecting the very outage this review is about. `check_reproducibility` and `check_training_pipeline` both pass (they are slow, >60s; an early run of mine timed them out and briefly looked like failures). |
| Quota deadlock | coverage grows 8 → 64 over 8 runs where it was previously frozen |
| Cost reduction | 50 → 25 requests for a 25-ticker run |
| JSON-body quota path | now classified `quota_exceeded`, stops after 1 call |

One existing test asserted the old deadlock (`assertFalse(cursor.exists())` —
"the cursor must NOT advance"). It was rewritten to pin the corrected contract,
with the measurement explaining why, plus five new tests covering the budget,
the no-progress nudge, cache reuse, date keying, and staleness ordering.

### 4.1 Sustainable rate

```
OLD: batch 40, events refetch    cost/day=80  sweep=1.9d  fits 50/12h=NO
NEW: batch 25, snapshots reused  cost/day=25  sweep=3.0d  fits 50/12h=YES
NEW x2 runs/day                  cost/day=50  sweep=1.5d  fits 50/12h=YES
```

The binding constraint is not the daily ceiling — it is that a full sweep must
finish inside the 7-day lookback, or the oldest ticker's news expires before its
turn comes round again. Batch 25 sweeps in 3.0 days with 50% headroom.

---

## 5. ML impact

**Direct feature dependency is small.** 2 of 40 registered features
(`news_sentiment_score`, `sentiment_score`), both `null_policy=exclude`.
`contextual_feature_surface` drops a non-OK contract entirely rather than
emitting a neutral value, so training rows lose a column instead of gaining a
fabricated one. Compare the fundamentals trap documented in
`core/training_features.py`, where absent data *did* become
`balance_sheet_quality = 10.0` for every company. News does not have that
failure mode.

**Event memory is where the loss bites.** All 361 records are `observed`, so
there is no inferred fallback in the store:

| Metric | Value |
|--------|-------|
| Tickers with any memory | 26 / 75 (34.7%) |
| Tickers with ≥5 analogs (`EVENT_MEMORY_MIN_ANALOGS`) | 18 / 75 (24.0%) |
| Usable (ticker, event_type) cells | 16 of 60 populated |
| Newest event | 2026-10-02 (4 days stale) |
| Business days with no news, last 30 | 22 |
| Of those, past the 7-day window | **16, permanently unrecoverable** |

So three quarters of the universe cannot support a conditional event forecast
at all, and the system reports `INSUFFICIENT` rather than guessing — correct
behaviour, but it means the feature is mostly dark.

**A separate finding, not caused by the quota:** 83.9% of events classify as
`event_type: "other"` (303 of 361). Analogs are matched on `event_type`, so an
`other` bucket this dominant limits specificity no matter how much news is
collected. Fixing the quota will not fix this; the classifier taxonomy in
`NEWS_CATEGORY_PATTERNS` is worth a separate look.

**Degradation risks, in priority order:**

1. *Silent thinning.* Training continues with fewer features and fewer analogs
   and does not announce it. The new coverage metrics are the mitigation.
2. *Survivorship in the memory store.* Memory concentrates in whichever tickers
   the frozen cursor happened to cover, so analogs are drawn from an
   accidentally-selected subset. Staleness-first ordering reduces this.
3. *Unrecoverable gaps.* 16 business days are already past the window. Nothing
   can recover them; the metrics now at least make the edge visible.

---

## 6. Architecture: where collection should run

The user's original question was how to keep collecting with the laptop off.

**The scheduled task did not survive sleep as it was configured:** `WakeToRun`
was `False` (now fixed — see *What was done* below). `StartWhenAvailable: True`
only catches up on the next boot, and by then the 7-day window has eaten the
oldest day.

| Option | Cost | Verdict |
|--------|------|---------|
| `WakeToRun: True` on the existing task | free | **Done.** One setting, no new infrastructure. Wakes from sleep, not from shutdown. |
| GitHub Actions `schedule` | free tier | **Viable, but needs a state decision first** — see below. Also note scheduled workflows are throttled under load and disabled after 60 days of repo inactivity, so it is not a guarantee. |
| Always-on host (Pi, cheap VPS) | ~free / ~$5mo | Most robust; closest to current behaviour. |

**The blocker for GitHub Actions is state, and it is already documented in
`.gitignore`:** the cursor, the collection report, the event memory and the raw
news ledger are all deliberately untracked, because news writes ~208 KB per
ticker per day — **~15.7 MB/day, ~3.9 GB/year.** Git never forgets, so
committing results back is irreversible and would make the repo unusable within
months. `data/` is already 149 MB, 130 MB of it `data/raw`.

So "run it in Actions and commit the output" is not a small change — it
contradicts a constraint the project has already reasoned about and written
down.

**Recommended split if you go cloud:** keep the append-only raw ledger off git;
put the *small, mutable* state (cursor, collection report) in a place both
environments can read — the simplest free option being an Actions cache or a
dedicated private repo/branch holding only those few KB — and treat the raw news
store as a machine-local artifact with its own backup, exactly as the
`.gitignore` comment already prescribes. Note that one consequence stands either
way: a fresh clone can replay price/fundamentals/macro scores but **cannot**
replay a news-derived sentiment, because those articles live only on the
machine that fetched them.

### What was done

**1. `WakeToRun` is now enabled** on the registered task (verified:
`WakeToRun=True`, next run 2026-10-06 22:00). The installer script
[`scripts/install_daily_task.ps1`](../scripts/install_daily_task.ps1) was also
fixed — it carried the comment "Wake the machine if asleep" but never passed
`-WakeToRun`, so re-running the installer would have silently reverted the fix.
The comment described an intent the code never had.

**2. The workflow is written** —
[`.github/workflows/daily-collect.yml`](../.github/workflows/daily-collect.yml) —
using the state split above rather than the commit-back approach:

| State | Size | Where it goes |
|-------|------|---------------|
| `collect_cursor.json` | 60 B | Actions cache (`restore-keys` so the cursor advances across days) |
| `collection_report.jsonl` | 5.7 KB | Actions cache |
| `data/raw/newsapi_news/` | 1.1 MB, +15.7 MB/day | 90-day artifact, **never committed** |

Design points worth noting: the cache is saved with `if: always()` because a run
stopped early on quota still advanced the cursor, and discarding that would
recreate the deadlock; `concurrency` does **not** cancel in-progress runs,
because a cancelled run may already have fetched perishable news; and the
collector's exit 1 is tolerated while the *monitor's* verdict fails the job,
since perishable loss is the expected free-tier steady state but an auth failure
is not.

It is committed but **not enabled until you add the `NEWS_PROVIDER_API_KEY`
repository secret**. Two things to decide before you do:

- **It will share the quota with the local task.** Two collectors against one
  50/12h window halves what each can fetch. Measured: both at batch 25 spend
  **exactly 50/day — zero headroom**, which is how this outage started. If you
  keep both permanently, set `COLLECT_NEWS_BATCH_SIZE = 12` (24/day combined,
  3.1-day sweep, still inside the 7-day window with >20% headroom). Otherwise
  retire the local task once you trust the cloud one.

  **This is now enforced rather than documented.** `COLLECT_NEWS_ACTIVE_COLLECTORS`
  defaults to 1; set it to 2 when you enable the workflow and the validator
  checks the *combined* spend against an 80% headroom cap:

      2 collector(s) at batch 25 spend 50 requests, above the 40 that keeps
      20% of the provider's 50-per-rolling-window allowance free for retries.
      Lower COLLECT_NEWS_BATCH_SIZE, or run fewer collectors

  A config that would recreate this outage now fails at import rather than at
  22:00 three days later. Batch 25 stays the shipped default because it is
  correct for one collector.
- **The reproducibility consequence is real.** A cloud-collected day's news-derived
  score is replayable only if you download the artifact within 90 days. That is
  the same limitation `.gitignore` already accepts for this machine, not a new
  one — but in the cloud it expires on a timer rather than sitting on a disk.

---

## 6b. Training pipeline: four faults, fixed

These were found while writing the operator runbook, are unrelated to the quota
work, and were fixed separately on request.

| # | Fault | Cause | Fix |
|---|---|---|---|
| 1 | `embargo (60) must be >= the max label horizon (252)` | The leakage gate took the max over **every** declared horizon. A dataset trains ONE horizon (default 20d), so a 20d fit was held to a 252-session embargo. | `build_walk_forward_folds` takes `target_horizon`; `train_baseline` passes `dataset.target_horizon`. Omitting it keeps the old conservative behaviour. |
| 2 | `KeyError: 'regime_probability_proxy'` | `feature_names` was built as a **union** of row keys, so a feature present on some rows and absent from others named a column the matrix build then demanded. | Intersection instead of union, plus a `partial_features` field recording anything dropped — so an intermittent producer is visible rather than fatal. |
| 3 | `label_version 'outcome-label-v1' != 'outcome-label-v2'` | `getattr(dataset, "label_version", "outcome-label-v1")` — `TrainingDataset` has no such attribute, so the stale literal was the **only** path, not a fallback. | Read `OUTCOME_LABEL_VERSION` from config, which is what the rows are validated against. |
| 4 | `history of 288 sessions is too short for fold_sessions=252` | `establish_baseline.py` passed no geometry, inheriting backtest defaults needing 882 rows, while its own arguments cap it at ~300. | Geometry sized to the dataset (fold 60 / holdout 40), embargo derived from the label horizon so the out-of-sample claim stays honest. |

Fault 1 alone unblocked `train.py`. Fault 2 was latent rather than active once
the embargo was fixed, but it is a real hazard and the invariant the code already
claimed in a comment is now enforced.

**Verified after the fixes:** `train.py --all` trains all 8 estimators over 5
folds; `establish_baseline.py` runs end to end and registers an incumbent
(`historical_mean`, directional accuracy 0.708, trial `d033e40c78df`).

The leakage rule is unchanged and still tested: a 120d or 252d label with a
60-session embargo is still refused.

## 7. Risk assessment

| Risk | Severity | Status |
|------|----------|--------|
| Quota deadlock recurring | was critical | Fixed; regression-tested; config invariant blocks the arithmetic |
| Auth failure read as a quota day | high | Fixed; alerts CRITICAL and says it will not self-heal |
| Permanent news loss while nobody notices | high | Metrics surface it; **16 days already lost** |
| Alert fatigue from 77 duplicate messages | medium | Per-ticker reasons; quota still deliberately un-alerted |
| Digest sender still emits old text | **open** | Sender is not in the repo, on disk, or in git history. Correct text is now callable via `--explain-tickers [--json]`; the sender must be pointed at it. |
| `event_type: other` at 83.9% | medium | Unaddressed; pre-existing, independent of the quota |
| Free tier insufficient for 75 tickers daily | medium | Sweep is 3.0 days by design. A daily sweep needs a paid tier or a smaller universe. |
| Two collectors silently halving each other's quota | was high | Blocked by `COLLECT_NEWS_ACTIVE_COLLECTORS` + 80% headroom invariant |

---

## 8. Implementation plan

**P0 — done in this change**
1. Stop the events stage refetching (halves cost).
2. Advance the cursor by work served (breaks the deadlock).
3. Classify failures by kind, not by substring.
4. Batch 25 + enforced budget 45 + cost invariant in the validator.

**P0 — not code, needs the operator**
5. ~~Set `WakeToRun: True`~~ — **done**, and fixed in the installer so it stays.
6. Point the digest sender at `monitor_collection.py --explain-tickers --json`.
   Until then the emails keep their misleading text. The fix is reachable from
   the command line; it cannot be wired from here because the sender is not on
   this machine.
7. Add the `NEWS_PROVIDER_API_KEY` secret to enable the workflow, having decided
   the batch-size question above.

**P1 — next**
8. Alert on sustained 0% coverage (distinct from "the collector did not run";
   today it ran fine and captured nothing).
9. Revisit `NEWS_CATEGORY_PATTERNS`: 83.9% `other` caps analog specificity.

**P2 — later**
10. Persist per-ticker last-success in a small index rather than rescanning the
    report ledger each run (fine at today's size, not at a year's).
11. Consider a second free news provider for redundancy; the typed taxonomy
    makes failover tractable.
12. Backfill inferred memories for the 49 uncovered tickers so analogs do not
    depend on which tickers the cursor happened to reach.
