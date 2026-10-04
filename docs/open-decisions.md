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

## 8. `test_current_and_long_term_scores_diverge_by_regime` is network-flaky — CLOSED

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

**FIXED (2026-09-27), by the preferred route.** The test now replays from
`data/raw/yahoo_finance_chart/`, which is tracked on purpose as the W6
provenance ledger, via the `rebuild_price_frame` helper that already existed.
All five tickers resolve from `*_15y_1d` records spanning 2011-09-19 to
2026-09-18, so 2024-01-02 is covered on a fresh clone with no network.

Verified rather than assumed:

* **The failure mode was reproduced first.** Feeding every ticker the same
  frame collapses all five scores to a single value (5.26), which is exactly
  the `len(set(...)) > 1` failure. The patch point matters: several modules
  bind `fetch_price_history` at import, so `agents.market_data_agent` is the
  one that has to be patched.
* **`build_score` reaches TWO providers, not one.** Replaying only the Yahoo
  price frames still hit `www.alphavantage.co` for fundamentals. This was
  caught only by testing against a *fresh clone*, which has no
  `data/*.parquet` cache — the working directory passed while the clone
  failed, which is precisely the gap open item 6 exists to close. Fundamentals
  now come from an offline fixture.
* **Holding fundamentals constant does not weaken the assertion.** With one
  snapshot given to all five tickers the current scores are still five distinct
  values (5.26 / 4.45 / 7.91 / 4.53 / 7.67), and the long-term scores likewise.
  Price alone drives the separation the test measures. The real snapshot is a
  `manual_fallback_contract` with all-null metrics anyway, since no API key is
  configured.
* **The fix was proved against a blocked network on a clean clone.** With
  `socket.connect`, `create_connection` and `getaddrinfo` all raising and zero
  parquet files present, both tests pass; the old version fails with a
  `ValueError` out of `agents/market_data_agent.py:118`.
* **The collapse is now its own regression test**
  (`test_identical_frames_collapse_the_scores`), so the degraded-provider
  behaviour is pinned rather than merely avoided.

---

## 9. A sabotage probe needs a plausible wrong answer available — OPEN

**Raised:** Sprint A (2026-09-23). **Owner:** unassigned.

MEASURED across Sprint A: **six** sabotages passed a gate that was supposed to
catch them, and in every case the gate was fine — the PROBE was too weak to
exercise the branch it targeted.

| task | sabotage that wrongly passed | why the probe could not decide |
|---|---|---|
| A2 | band-cross branch removed | every scenario also changed the binding factor or cleared the magnitude bar |
| A3 | (contract-check bug, different shape) | substring guard flagged the field that *declares* magnitude unused |
| A4 | computability guard removed | every uncomputable scenario already had `label=None` |
| A6 | confidence read as "first numeric value" | the probe was `{"band": "HIGH"}` — no numbers to guess from |
| A7 | de-escalation rule removed | probe sat inside the cooldown, which already suppressed |
| A7 | absent severity ranked `0` | the gate never probed an appearing/vanishing severity |

**The pattern.** A sabotage only proves something when a plausible WRONG answer
is reachable in the scenario. `{"band": "HIGH"}` cannot catch "guess a number"
because there is no number to guess; `sessions_since=1` cannot catch "ignore
de-escalation" because the cooldown decides first. A passing sabotage is
therefore ambiguous by default — it means either the guard works or the probe
is inert, and those are not the same result.

**Why it matters.** Five of the six were found only because each sabotage was
re-checked for reachability before being trusted. Without that habit, all five
would have been recorded as "guard verified" while the guard was untested.

**What would settle it:** a convention that every sabotage is accompanied by a
positive control — a paired scenario that FAILS when the guard is removed and
PASSES when restored — so an inert probe is visible rather than silent. A7 does
this ad hoc (`unchanged_old` beside `relieved`); making it a rule would cost
little and remove the ambiguity.

---

## 10. F7's structured assessment has no declared consumer contract — OPEN

**Raised:** Sprint A (2026-09-23). **Owner:** unassigned.

F7 stores its WHOLE assessment mapping in a field whose name refers to a
scalar. Two separate tasks hit this independently and each had to special-case
it: D1's renderer dumped a ~4,000-character mapping into a table cell, and A6
raised on it before learning to read the scalar from the field that names the
quantity.

**Why it matters.** The mapping carries several unrelated floats
(`binding_value`, `weighted_sum`, every factor score), so a consumer that
guesses — "take the first number" — silently thresholds on a factor score
instead of the confidence. A6's gate now catches exactly that, but only for A6.
Every future consumer re-derives the same handling, and the third one may guess.

**What would settle it:** either a documented accessor F7 exports (so the
scalar is read one way everywhere), or a named field carrying the scalar
alongside the assessment. Two independent consumers hitting the same shape in
one sprint suggests a third is coming.

---

## 11. Outbound SMTP is intercepted on the operator network — DECISION

**Raised:** 2026-10-04. **Owner:** operator.

The alert channel was built and verified this session: `core/alert_delivery.py`
(+40 tests) and `scripts/verify_alert_email.py`. The credential is set and
correctly shaped. **The channel still cannot deliver**, and the cause is the
network, not the setup.

MEASURED 2026-10-04:

```
tcp connect smtp.gmail.com:587   OPEN  (142.251.127.109)
tcp connect smtp.gmail.com:465   OPEN
tcp connect smtp.gmail.com:25    OPEN
SMTP greeting banner on 587      NONE within 15s
SMTP greeting banner on 465      b''   (closed immediately)
HTTPS to gmail.googleapis.com    200   (port 443 unimpeded)
Windows proxy configured         no (ProxyEnable=0)
```

The TCP handshake completes and the session is then swallowed. A refused
connection would be a firewall; a completed connect with no banner is outbound
SMTP **interception** — routine on an enterprise-managed machine, which this is.

**What this rules out.** The credential is never reached, so the timeout says
nothing about whether it is valid. This is the first thing the `failed` vs
`auth_failed` status split bought: the failure is attributable without guessing.

**What would settle it:** a channel on port 443, since no SMTP channel will
work on this network whatever the provider. Telegram's bot API is the
recommendation — no OAuth flow, no business verification, and a real phone
push, which was the original ask. The Gmail REST API also rides 443 but needs
an OAuth flow for a benefit nobody asked for.

**Not a defect in the transport.** `send_alert_email` reported the failure
exactly as designed, returned a distinct status, and never raised — the daily
collector would have kept collecting. The module stays, behind `smtp_factory`,
so a 443 channel substitutes for it rather than replacing it.

## 12. Nothing on the live path sends an alert — OPEN

**Raised:** 2026-10-04. **Owner:** unassigned.

MEASURED 2026-10-04, for each of the seven alert evaluators, the complete list
of non-test importers:

```
forecast_alert            core/config.py  scripts/check_forecast_alert.py
thesis_alert              core/config.py  scripts/check_thesis_alert.py
regime_alert              core/config.py  scripts/check_regime_alert.py
event_impact_alert        core/config.py  scripts/check_event_impact_alert.py
confidence_alert          core/config.py  scripts/check_confidence_alert.py
forecast_threshold_alert                  scripts/check_forecast_threshold_alert.py
alert_suppression         core/config.py  scripts/check_alert_suppression.py
```

Every one is imported only by its own CI gate. `scripts/daily_collect.py` and
`scripts/run_forecasts.py` do not contain the string "alert" at all.

**Why it matters.** Sprint A built six alert evaluators and a suppression gate
that governs *delivery* — and A7's own prose says a vetoed alert "is still
delivered, marked non-actionable". Nothing was ever delivered, because nothing
called them. The gates verify the evaluators are correct, which is a different
claim from the system alerting anybody.

**What would settle it:** one wiring task — `run_forecasts.py` calls the
evaluators, passes survivors through `alert_suppression`, and hands the result
to a delivery channel. Blocked on item 11 for the channel, not for the wiring.

## 13. `install_daily_task.ps1` points at a venv that does not exist — CLOSED

**Raised:** 2026-10-04. **Owner:** unassigned.

The installer resolves `$repo\.venv\Scripts\python.exe` and falls back to bare
`python` when absent. The venv in this repo is `venv`, with no leading dot, so
**the fallback always wins**.

MEASURED 2026-10-04:

```
.venv                 does not exist
venv                  exists
bare python           3.12.10, pandas 3.0.5, numpy 2.5.2, sklearn MISSING
venv  python          3.12.10, pandas 3.0.5, numpy 2.5.2, sklearn 1.9.1
```

The fallback is the dangerous kind: bare python has pandas, so collection would
appear to work, while anything touching the model path fails on a missing
sklearn — under a scheduler, at 22:00, into a log nobody reads.

~~**What would settle it:** resolve `venv` as well as `.venv`, and **refuse to
register the task** when neither has the required imports.~~ **FIXED
2026-10-04.** The installer now tries both spellings and runs an import probe
(pandas, numpy, sklearn, pyarrow) against the resolved interpreter, exiting 1
with the missing module named rather than registering a task that would fail
nightly. VERIFIED both ways: the probe returns OK on `venv` and
`MISSING:sklearn` on the bare interpreter.

The probe itself had to be fixed before it worked — written as
`import importlib` it raised `AttributeError: module 'importlib' has no
attribute 'util'` on EVERY interpreter, which would have blocked the install
entirely instead of guarding it. A guard that has never been watched failing
is not a guard; this one was tested in both directions before being trusted
(the open-item-9 pattern, in a shell script this time).


---

## 14. The collector reports news OK on a dry run with no key — OPEN

**Raised:** 2026-10-04. **Owner:** unassigned.

MEASURED 2026-10-04, the same collector invoked two ways, on a machine with
`NEWS_PROVIDER_API_KEY` unset:

```
--dry-run   news  OK      PERISHABLE  {'attempted': 40, 'ok': 40, ...}
real run    news  FAILED  PERISHABLE  {'attempted': 40, 'unavailable': [40 tickers]}
```

The dry run reports `ok: 40` for a source that **cannot** succeed, because it
counts tickers it WOULD attempt rather than probing whether the provider is
reachable. The real run correctly reports `FAILED` and exits 1.

**Why it matters.** A dry run exists to be trusted before committing to a
scheduled job. This one says the perishable source is fine in precisely the
configuration where it is guaranteed to lose the day — the same
"looks-fine-until-22:00" shape as item 13.

**What would settle it:** the dry run resolves the credential (without
fetching) and reports `news UNAVAILABLE (no key)` rather than `OK`. Checking a
key is set costs nothing and is exactly what the dry run is for.

## 15. Six references to a variable that does not exist — CLOSED

**Raised:** 2026-10-04. **Fixed:** same day.

MEASURED: five scripts referenced `NEWSAPI_KEY`; the variable the code
actually reads is `NEWS_PROVIDER_API_KEY` (`core/config.py:265`). The worst was
`scripts/daily_collect.py:467`, the operator-facing instruction printed on
every failed collection:

```
News needs NEWSAPI_KEY. Until it is set, no OBSERVED
```

So the one message telling the operator how to fix the perishable-data loss
named a variable nothing reads. This cost real time: the operator set
`NEWSAPI_KEY` in a previous session and news collection stayed dead.

Corrected in `build_event_memory.py`, `check_data_coverage.py` and
`daily_collect.py`.


---

## 16. The first OBSERVED event memories exist — and carry no gradeable horizon yet

**Raised:** 2026-10-04. **Status:** informational; closes itself with time.

`NEWS_PROVIDER_API_KEY` was set and the first news collection ran. MEASURED
2026-10-04, immediately after:

```
raw news days in the ledger                 1  (2026-10-04, was 0 EVER)
event memories written                    361
provenance                           observed  (was 100% price-inferred)
inference_method                             "" (not quarterly_volume_cadence)
distinct tickers                            26
entity_resolution_method   none 243 | ticker 63 | alias 34
                           ambiguous 18 | legal_name 3
```

**The horizons are the thing to understand.** Of 361 memories, **318 carry
only a 1d response** and 43 carry 1d+5d. None reach 20d or 60d:

```
published dates   2026-09-27 .. 2026-10-02
```

This is not a defect — a 20d outcome needs 20 sessions *after* the event, and
the newest news is days old, so those horizons **cannot** exist yet by
construction. They accrue as sessions pass, which is the argument for
scheduling rather than an argument against the data.

**What it means for grading today.** `EVENT_MEMORY_MIN_ANALOGS = 5` at the 20d
horizon still has nothing to work from. The 780-memory price-inferred store
previously reported a median 20d move; these 361 cannot contribute to that
number for about a month.

**Watch `entity_resolution_method: none` at 243 of 361 (67%).** That is the
figure the Finnhub swap was meant to improve — Finnhub returns a provider-side
`related` ticker field rather than requiring keyword resolution. Worth
re-measuring once more days accrue, since one day's sample over 26 tickers is
thin evidence for a provider decision.

## 17. X4's "no event memories" assertion expired the day news arrived — CLOSED

**Raised:** 2026-10-04. **Fixed:** same day.

`tests/test_event_robustness.py::test_there_are_no_event_memories` asserted
`len(load_memories()) == 0` and failed (`361 != 0`) on the first collection
run. **The test was right to fail** — it was X4's blocking measurement, and
the premise expired.

X4 named TWO blockers and said explicitly that fixing only the first "still
leaves nothing joinable". MEASURED 2026-10-04, the second is untouched across
all 8 shipped runs:

```
fold keys: actuals, fold_id, metrics, predictions, train_end_time,
           train_rows, validation_rows, validation_start_time
event_id / source / ticker / timestamp:  ABSENT
```

So X4 still reports NOT_EVALUATED, for the surviving half of its original
reason. The test now asserts **the join key is absent** rather than that the
data is absent — the condition that actually gates the gate, and one that
fires the day folds start carrying attribution.

**The replacement was sabotage-tested before being trusted:** injecting
`event_ids` into each fold makes it fail with 8 subtest failures naming X4,
so it is reachable rather than an inert probe (open item 9's convention,
applied).

A second test was added asserting an on-disk memory declares `provenance` and
`entity_resolution_method` — deliberately NOT pinned to a count, which grows
daily and would fail every collection.


---

## 18. Polling cadence is a QUOTA decision, not a scheduling one — DECISION

**Raised:** 2026-10-04. **Owner:** operator. **Supersedes part of item 7.**

The operator asked whether collection could run every 2 hours instead of once
at 22:00, so alerting behaves like monitoring. MEASURED the same day, by
exhausting the quota accidentally:

```
HTTP 429  code: rateLimited
"Developer accounts are limited to 100 requests over a 24 hour period
 (50 requests available every 12 hours)"
```

Two facts that change the answer:

1. The limit is **100/24h enforced as 50/12h**, and it is a **ROLLING**
   window — not a midnight reset. A schedule cannot "start fresh" at 00:00.
2. 75 tickers are news-eligible, so **one full sweep costs 75 of the 100**.

The cadence arithmetic, against `COLLECT_NEWS_BATCH_SIZE = 40`:

| schedule | requests/run | full sweep | verdict |
|---|---|---|---|
| every 2h (12 runs) | 8 | 18h | starves every run |
| every 2h, market hours (6 runs) | 16 | 9h | thin but viable |
| 2 runs/day (12h-aligned) | 50 | 18h | matches the window |
| 1 run/day (today) | 40–75 | 1–2 days | current |

**More frequent polling does not mean fresher data on the free tier.** The
quota is the binding constraint, so 12 runs/day means each run sees 8 tickers
and any given ticker is polled less often than it is today. Frequency and
coverage trade directly against each other.

**What the operator's filtering instinct gets right.** A relevance-ranked
subset is exactly the mechanism that makes frequent polling coherent — poll
10 high-signal names every 2h, sweep the rest daily. The parts already exist:
`--tickers`, `--limit`, `COLLECT_NEWS_BATCH_SIZE`, the rotation cursor, and
`COLLECT_NEWS_TRACK_ANYWAY` as precedent for a named exception list. What does
NOT exist is a *defensible ranking* — and picking the "most relevant" tickers
by judgement would be the survivorship-bias-by-construction trap that register
item 3 refuses for breadth features.

**What would settle it:** either
(a) accept 1–2 runs/day on the free tier and keep 22:00, or
(b) make the Finnhub swap first (item 16), whose free tier is ~86,000
    calls/day — which makes 2-hourly polling of the FULL universe possible
    and removes the ranking problem entirely rather than solving it.

(b) is the recommendation: the quota, not the schedule, is what blocks
monitoring-grade alerting, and Finnhub removes the quota.

**A prerequisite either way:** item 12 — nothing on the live path calls an
alert evaluator, so no polling frequency produces an alert yet.

## 19. Quota exhaustion was handled correctly — CLOSED (verified, no change)

**Raised:** 2026-10-04. **Verified:** same day.

The second manual run hit the 429 mid-sweep. MEASURED immediately after:

```
cursor                     5   (NOT advanced past the failed tickers)
event_memory.jsonl       361   (intact; the prior run's memories survived)
news ledger days           1   (2026-10-04 preserved)
exit code                  1   (fail-loud: perishable loss reported)
report                     quota_exhausted: True
```

The collector detected the 429, reported `PROVIDER QUOTA EXHAUSTED`,
**suppressed the cursor advance** so the unfetched tickers are retried rather
than skipped, and left prior data untouched. Prices and fundamentals still
captured for all 77 tickers — fail-soft per source held.

Recorded because it is the first time the quota path ran against a real 429
rather than a test, and it behaved as designed. No change needed.


---

## Closed

- ~~**Live runs must use today's chart state.**~~ Fixed in `ad75f48`. The
  memory-derived state was a median 144 days stale and retrieved analog sets
  overlapping the live ones by a Jaccard of 0.205, calling VOO and CIBR
  bearish while both were bullish.
