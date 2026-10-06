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

## 4. The source → outcome join does not exist — CLOSED (2026-10-04)

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

**CLOSED 2026-10-04 by steps 1 and 2** (`6dcfddf`, and the commit that
follows it). Both halves the entry asked for now exist:

```
                              before      after
Event.source                  provider    THE OUTLET
EventMemory.source            (absent)    THE OUTLET
articles carrying a ticker    0 of 1,637  353 of 1,637  (21.6%)
```

Resolution is done at ingestion with E2's resolver:

```
none         1,169  (71.4%)
ticker         265  (16.2%)
ambiguous      115  (7.0%)
alias           72  (4.4%)
legal_name      14  (0.9%)
executive        2  (0.1%)
```

L4 can now MEASURE an outlet instead of asserting a prior — Biztoc.com
(155 joinable articles), Rlsbb.cc (44) and Pypi.org (16) are the first
outlets whose reliability is testable rather than assumed. The estimator
needed no change: shrinkage at k=20 already degrades to the global rate
until a cell earns its own.

Note what is NOT closed: outcomes have to ACCRUE before a rate means
anything. See item 16 — the memories are days old, so no 20d horizon
exists yet.

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

## 20. Many sources is the goal — and the outlet is dropped before it can be judged

**Raised:** 2026-10-04. **Owner:** operator + next sprint.
**Supersedes the "what it would take" clause of item 4.**

The operator's stated goal: **as many different news sources as possible, then
filter and investigate them** — which is what L4's source-reliability
estimator was built for. MEASURED 2026-10-04 against the first real news day,
the breadth already arrived and the machinery cannot see it.

**The corpus is already broad, and already mostly noise:**

```
distinct source_name values            369  (was 50 on 2026-09-21)
articles in one day                  1,637
from a recognisable financial outlet   114  (7.0%)

top outlets by volume
  265  Biztoc.com            aggregator, republishes others
   96  Pypi.org              the Python package index
   62  The Times of India
   57  PRNewswire
   48  Rlsbb.cc              filesharing site
   32  Bringatrailer.com     classic car auctions
    8  Biblegateway.com
```

A real headline from the ledger, stored as market news:

> "Xiaomi Redmi Note 17 Pro 5G 6GB RAM/256GB Storage $406.80 @ Mobileciti"
> — Ozbargain.com.au

**Three measured blockers, in the order they bite:**

```
1. source_quality populated      0 of 1,637  (100% None)
2. ticker populated              0 of 1,637  (100% None)
3. Event.source                  "newsapi_news" at 7 hardcoded call sites
```

**(1) means every outlet is weighted identically.** `source_weight()` treats a
`None` quality as the neutral factor **1.0**, so Bringatrailer.com and CNBC
carry the same weight into every score. Documented as neutral semantics, which
is correct in isolation and wrong at 369 outlets.

**(2) is item 4, unchanged at 1,637 articles.** No article joins to a price
outcome, so no outlet can ever be credited or blamed.

**(3) is the newly located root cause.** `core/news_adapter.py` hardcodes
`NEWS_SOURCE_ID = "newsapi_news"` and passes it as `source_id` at seven call
sites. The per-article `source_name` IS captured in the raw ledger and IS
read by `_normalize_provider_payload`, but it never reaches an `Event` —
`Event.source` therefore records the PROVIDER, not the OUTLET.

**Why that last one matters most.** `Event` already HAS `source` and
`source_quality` fields, and `event_from_article` already reads
`article["source_weight"]`. The plumbing exists end to end; one constant
overwrites the outlet on the way through. L4 is not missing a feature — it is
being handed the provider's name 1,637 times and correctly reporting
REGISTRY_PRIOR because it has one "source" with no variation to learn from.

**What breadth actually requires, in dependency order:**

1. **Carry the outlet.** Pass `source_name` as `Event.source` and keep the
   provider in a separate `provider` field. Cheap, unblocks everything below.
2. **Resolve the ticker at ingestion** (item 4's requirement). Closes the
   source → outcome join, so an outlet's record can be MEASURED instead of
   asserted.
3. **Let L4 learn.** It already shrinks per-outlet rates toward a global rate
   with k=20 pseudo-counts and reports NO_EVIDENCE / REGISTRY_PRIOR / GLOBAL /
   CONDITIONAL. Once (1) and (2) land it starts learning with no new estimator.
4. **Add providers, not filters.** Each new provider widens the corpus; L4
   then demotes the noise FROM EVIDENCE rather than from a hand-written
   allowlist.

**Why not a quality allowlist.** Ranking 369 outlets by judgement — and the
list grows daily — is the survivorship-bias-by-construction trap item 3
refuses for breadth features, and it discards the only thing that makes the
noise informative: whether an outlet's stories actually precede price moves.
`SOURCE_RELIABILITY_ABSENT_IS_ZERO = False` already encodes this: an
unmeasured outlet and a useless one are different facts.

**Finnhub's role is narrower than it looked.** It fixes (2) at the source via
a provider-side `related` ticker field, and removes the 100-request quota. It
does NOT fix (1) or (3) — those are this repo's code, and no provider choice
substitutes for carrying the outlet through.


---

## 21. A length guard cannot separate a ticker from a word — DECISION

**Raised:** 2026-10-04. **Owner:** operator.

`ENTITY_AMBIGUOUS_TICKER_MAX_LENGTH = 2` makes short tickers match by name
instead of by symbol, which correctly refuses `KO`, `BE` and `V`. ARM is three
characters, so it passes — and MEASURED 2026-10-04 against the exact case the
operator first described:

```
resolve_entity("1968 Corvette with a new ARM rest", "ARM")
  -> matched=True  method=ticker  confidence=1.00
```

A classic-car listing resolves to the ARM Holdings ticker at MAXIMUM
confidence.

**Raising the guard to 3 is the wrong fix, measured.** The universe holds 17
three-letter tickers, and all 17 resolve correctly today on an ordinary
finance headline:

```
3-letter tickers in universe          17
also common English words              3   (ARM, CAT, NOW)
of those, already caught by name rule  1   (CAT)
legitimate matches broken by a len-3 guard  17
```

So a length-3 guard would break 17 working matches to fix 2. The instrument is
wrong, not the threshold: length does not distinguish a symbol from a word,
and the registry already knows the difference.

**What would settle it:** a per-entity `requires_name` flag on the handful of
tickers that collide with common words, so the registry carries the exception
with provenance (E2's "a registry, not a regex" rule) instead of a global
threshold punishing every short ticker. Three entries would cover the measured
cases.

**Not fixed in step 2** because it changes resolution for tickers beyond the
articles step 2 touches, and because `ARM` matching in caps is a different
failure from the uppercase bug step 2 fixed — that one destroyed case
information; this one has the case right and still cannot tell a symbol from a
noun.

## 22. Step 2's coverage is bounded by the entity registry — OPEN

**Raised:** 2026-10-04. **Owner:** unassigned.

E2 refuses a ticker it cannot verify, with a reason:

```
"TEST is not in the entity registry, so a mention of it cannot be
 verified — add it rather than guessing"
```

That is the right behaviour, and it bounds step 2: the registry holds **77
entities**, so resolution coverage can never exceed the universe the registry
describes. The 71.4% `none` rate is partly this and partly genuine noise, and
those two causes are NOT currently separable in the report.

**What would settle it:** split `none` into `unregistered_ticker` and
`no_match_in_text`. The first is a registry gap (fixable by adding an entry);
the second is a real non-match. Collapsing them repeats exactly the mistake E2
was built to fix — the entry's own docstring says a failed resolution must not
look like a correct exclusion.


---

## 23. A phone app is the wrong shape for phone monitoring — CLOSED (decided)

**Raised:** 2026-10-04. **Decided:** same day, by the operator.

The operator asked for a phone app that would "start the server", because
they are not always with the laptop but always with the phone.

**Why an app cannot do that.** The scoring engine needs pandas, numpy,
sklearn and pyarrow against a 143MB local store. None of that runs on a
phone, so any "app" is a thin client talking to a server that still has to
run somewhere — a hosting problem wearing an app costume.

The three real options, measured against the stated need:

| option | runs 24/7 | cost | on-demand score | work |
|---|---|---|---|---|
| push alerts to the phone | no (laptop) | free | no | one channel |
| cloud host + mobile web | yes | $5-10/mo | yes | secrets, auth, deploy, 143MB store |
| Tailscale to the laptop | no (laptop) | free | yes | bind address only |

**Decided: push alerts.** It delivers the actual requirement — knowing when
something happens while away from the laptop — with no server, no store and
no app. What it does NOT give is an on-demand score; the operator receives
what the last scheduled run computed.

Worth recording for later: `index.html` ALREADY carries
`<meta name="viewport">` and two media queries (900px, 640px), so the
existing UI is substantially mobile-ready if the cloud-host option is ever
revisited.

## 24. Telegram is blocked on the Aman network too — DECISION

**Raised:** 2026-10-04. **Owner:** operator. **Sibling of item 11.**

The Telegram channel was built as the answer to item 11 (SMTP intercepted).
It is blocked on the same network, by the same appliance, and the operator
will run it from their home network instead.

MEASURED 2026-10-04:

```
tcp api.telegram.org:443       OPEN  (149.154.166.110)
POST /bot<token>/getMe         HTTP 503, an HTML captive-portal page
api.telegram.org cert issuer   palo-decrypt.scp.co.il
gmail.googleapis.com issuer    Google Trust Services (WR2)
```

A certificate issued by an appliance rather than a public CA means TLS is
terminated and inspected locally. Google hosts present genuine certificates,
so this is **category-based filtering of messaging apps**, not a general
egress block.

**A LESSON ABOUT THE PROBE, not just the network.** Before building, a
reachability check reported `tcp 443 OPEN` and "the host answered", and that
was taken as proof the channel would work. It proved only that SOMETHING
answered — the appliance. A probe that cannot distinguish the real endpoint
from an interceptor is not a reachability probe. `diagnose_interception()`
now inspects the certificate issuer, which is the check that would have
caught this before a line of transport was written.

**What this does NOT mean.** The token is UNTESTED, not rejected: the
request never reached Telegram. That distinction is exactly why
`auth_failed` and `failed` are separate statuses, and the second time in one
session that the split paid for itself.

**Status: the channel code is complete and tested** (35 tests, transport
substituted, nothing touching the network). It needs to be RUN from a
network the operator controls. No further probing of the Aman network is to
be done — the operator has said so twice.


---

## 25. The raw ledger stores 99.6% redundant price data — DECISION

**Raised:** 2026-10-04. **Owner:** operator. **Bites:** in about a year.

MEASURED 2026-10-04, comparing the same request across two collection days:

```
AAPL_5y_1d on 2026-09-27     1,255 bars
AAPL_5y_1d on 2026-10-04     1,255 bars
overlapping bar_time         1,250
genuinely new                    5   (2026-09-28 .. 10-02)
REDUNDANCY                    99.6%
payload_sha256               DIFFERENT, so dedup-by-hash cannot catch it
```

Every run re-fetches 5 years of daily bars per ticker and appends the whole
series. The W6 store supersedes older versions **by payload hash**, and the
hash changes whenever a single new bar arrives, so each day's file is a
near-complete copy of the previous one.

**The cost, measured from the first full collection day:**

```
news  per business day    1.13 MB
price per business day    6.90 MB
TOTAL                     8.03 MB/day

1 month    0.16 GB
1 year     1.98 GB
5 years    9.88 GB
```

**Why this is a DECISION and not a defect.** The append-only ledger is W6's
provenance guarantee and the rebuild-from-raw proof depends on it. Trimming
it is a deliberate trade of provenance for disk, not a bug fix. Three
options, in increasing effort:

1. **Accept it.** 2 GB/year is tolerable on a laptop; revisit at 5 GB.
2. **Fetch incrementally** — request only bars since the last stored
   `bar_time`, keeping one full history per ticker plus daily deltas. Cuts
   price storage by ~99% and is the obvious fix, but it changes what
   "rebuild from raw" replays and needs the W6 proof re-verified.
3. **Compress the ledger** (gzip per day). ~10x for free, no semantic
   change, and `load_raw_records` would need to read both forms.

Option 3 is the cheapest real win and carries no provenance risk. Option 2
is the correct long-term answer and should not be done casually.

**Not urgent:** at 8 MB/day this is a next-quarter decision, not a
tomorrow one. Recorded now because the measurement only became possible
once collection actually ran for a full universe.

## 26. Four dashboard panels read fields the API never sends — OPEN

**Raised:** 2026-10-04. **Owner:** unassigned.

MEASURED 2026-10-04 by diffing the fields `index.html` reads against what
`orchestrate_score` returns:

```
UI expects, API never sends:
  - confidence_breakdown
  - insights              (bullish_signals / bearish_signals)
  - recommended_actions   (primary / options)
  - source_reliability    (sources / cross_validation)

API sends, UI ignores:
  + snapshot_hash
  + source_record_ids
  + summary
```

All four read with optional chaining and fall back to "None listed." or
"N/A", so nothing crashes — the panels are simply **permanently empty**. A
reader cannot tell "the system has no insights for this ticker" from "this
panel has never been wired", which is the same ambiguity E2 exists to
prevent in entity resolution.

**Why `source_reliability` is now the interesting one.** Steps 1 and 2
(2026-10-04) made the outlet and the resolved ticker flow through, so L4
can finally compute per-outlet reliability. The panel to display it already
exists and has existed all along.

**What would settle it:** either wire each field, or remove the panel and
say why. An empty panel that looks like a feature is worse than an absent
one. The three ignored API fields are also worth surfacing —
`snapshot_hash` and `source_record_ids` are exactly the provenance a reader
needs to trust a score.

## 27. A 7-day news window makes every-other-day polling lossless

**Raised:** 2026-10-04. **Status:** informational; settles part of item 18.

MEASURED 2026-10-04, the publish dates present in a single day's fetch:

```
2026-09-27   51
2026-09-28  114
2026-09-29  139
2026-09-30  175
2026-10-01  251
2026-10-02  501
2026-10-03  406
```

One request returns a **7-day history**, not just today's news
(`NEWS_LOOKBACK_DAYS = 7`). So a ticker polled on Monday still sees the
previous Thursday's articles.

**Why this matters for the cadence decision (item 18).** The earlier
recommendation of *daily* collection was argued from "news perishes at 7
days", which is true of the PROVIDER's window but not of a given run. With
40 of 75 tickers per run and a rotating cursor, every ticker is polled
every other day and **still loses nothing** — the gap only becomes lossy
above ~7 days.

That removes the urgency from daily-vs-alternate-day, and leaves the real
constraint where item 18 put it: the 100-request quota, not the schedule.
A gap of 7+ business days (like 2026-09-27 to 2026-10-04) IS lossy, which
is what makes scheduling matter at all.


---

## 28. What accuracy actually requires — the X-sprint findings, consolidated

**Raised:** 2026-10-04. **Owner:** operator + whoever owns the next sprint.
**This is the close-out the X sprint never got.**

X1-X8 measured more about this system's limits than any register entry
recorded, and those findings lived only in commit messages.

### The deciding measurement (X1)

Eight estimators, 20d horizon, 120 validation observations, 96-day embargo —
PIT-clean, so these are not leakage artefacts:

```
estimator            rmse      95% bootstrap CI     dir_acc
historical_mean   0.12164   [0.09741, 0.14757]      0.5500   <- BASELINE
random_forest     0.13192   [0.10390, 0.16231]      0.4583
gradient_boosting 0.14657   [0.11887, 0.17134]      0.4750
ridge             0.15769   [0.13414, 0.18321]      0.5000
momentum          0.17255   [0.15174, 0.19451]      0.5750
elastic_net       0.17850   [0.15316, 0.20415]      0.5333
mean_reversion    0.19152   [0.16585, 0.22028]      0.4250
logistic          0.19178   [0.16519, 0.22092]      0.4750
```

**Zero of seven learned estimators beat a no-feature baseline.** All eight
directional accuracies fall inside the +/-0.0895 coin-flip band at n=120. A
10,000-shuffle permutation test on the best (momentum, 0.5750) gives p=0.0625
— failing before any correction for having tested eight.

### Why this is a DATA problem, not a model problem

```
dataset rows                           406
features                                16  (ALL price-derived)
feature groups the task names             5
feature groups actually present           1  (technicals)       [X5]
horizons trained                          1  (20d of 5)         [X6]
walk-forward folds                        1                     [X1]
trials registered vs estimators trained 1 vs 8                  [X7]
regime recoverable from a fold           NO                     [X3]
event data joinable to a fold            NO                     [X4]
calibration measured out-of-sample       NO (in-sample ECE = 0)  [X2]
```

Four of five feature groups were never there to ablate. So the honest reading
of X1 is not "the models are bad" — it is **"16 price features on 406 rows
cannot predict 20-day returns, and nothing else has been tried yet."**

### The sample sizes accuracy would require

To distinguish a true directional edge from a coin flip at 95%:

```
true accuracy 0.55  ->    385 independent observations
true accuracy 0.54  ->    601
true accuracy 0.53  ->  1,068
true accuracy 0.52  ->  2,401
```

Rows are NOT independent. A 20d forward return sampled daily overlaps its
neighbour by 19 of 20 days:

```
             rows     ~independent 20d windows
1 month     1,617                      80
1 quarter   4,851                     242
1 year     19,404                     970
3 years    58,212                   2,910
```

Cross-sectional correlation makes it worse: 77 mostly-US-large-cap-tech names
move together, so 77 same-day rows are far fewer than 77 independent draws.

**Stated plainly: detecting a realistic edge (52-54%) needs on the order of
1-3 YEARS of accrued, matured outcomes.** No modelling choice shortens that.
This is the single most important fact for setting expectations about this
system.

### What to do, in dependency order

1. **Accrue data on a schedule** (item 7/18). Everything else is downstream.
2. **Add the four missing feature groups** (X5). Fundamentals and macro
   adapters already exist; news now resolves to tickers (item 4 closed). This
   is the highest-value modelling work BECAUSE it is the only untested
   hypothesis — price features have been tested and failed.
3. **Carry ticker, timestamp and regime onto each fold** (X3/X4). Cheap, and
   it unblocks two robustness gates that cannot run at all today.
4. **Train the other four horizons** (X6). 1d and 5d need far less data per
   conclusion than 20d, and multi-horizon agreement is itself evidence.
5. **Fix out-of-sample calibration** (X2). An in-sample ECE of 0.0000 is
   structurally incapable of being anything else.
6. **Register every trial** (X7). 1 registered vs 8 trained means the system
   cannot know how many tests it ran, so it cannot correct for them.

### What NOT to do

- **Do not add estimators.** X1 tested eight; the chance of a spurious winner
  among eight noise draws is 33.7%. A ninth worsens selection bias.
- **Do not open the sealed holdout** (X8). At 60 rows a fair coin posts 62.7%
  inside the band.
- **Do not report a directional accuracy without its band.** Every number in
  the X1 table is inside the noise, and the band is what says so.


---

## 29. The backfill tripled the memory store and changed no forecast — OPEN

**Raised:** 2026-10-06. **Owner:** unassigned.

`build_event_memory.py --mode backfill` was run to attack the "only 18 of 75
tickers reach 5 analogs" problem. It worked, on its own terms:

```
                           before   after
event_memory records          361    1,140   (+779 inferred)
tickers with ANY memory        26       73   of 75 eligible
tickers with >= 5 memories     18       67   of 75
usable (ticker, event_type)    16       79   cells
```

**And forecasting did not move at all: still 292 of 292 refused.**

The reason is that `EVENT_MEMORY_MIN_ANALOGS` is not the binding gate —
`EVENT_MEMORY_MIN_SIMILARITY` is. MEASURED across all 73 tickers holding a
memory, retrieving `earnings` analogs against today's live chart state:

```
similarity bar   tickers reaching 5 analogs   median analogs   tickers at zero
    0.70                 0 / 73                    0                 55
    0.60                24 / 73                    2                 19
    0.50                68 / 73                   21                  1
    0.40                72 / 73                   88                  0
    0.30                73 / 73                  236                  0
```

At the shipped bar of 0.70, **no ticker in the portfolio can produce a
conditioned forecast**, however many memories exist. The store is not the
constraint; chart-state similarity is.

**Why this is a DECISION and not a bug.** Lowering the bar buys analogs by
loosening what counts as comparable. At 0.50 the median ticker retrieves 21
analogs — but an analog that is only half-similar is a weaker claim about this
situation, and the whole point of the E6 design is that a "typical response"
drawn from loose matches describes the market rather than the event. The yield
table is the honest input to that trade-off; picking a number from it is a
judgement about how much similarity a published forecast should require.

**What would settle it:** measure forecast ACCURACY at each bar once outcomes
mature, and pick the bar that maximises it rather than the one that maximises
yield. That needs accrued matured outcomes, which is the same bottleneck as
everything else in section 28. Until then, 0.70 refusing everything is at least
a refusal rather than a weak claim dressed as a strong one.

**Not recommended:** lowering the bar to make the dashboard look populated.
Section 28's warning applies directly — a number inside the noise band, reported
without the band, is worse than no number.

## Closed

- ~~**Live runs must use today's chart state.**~~ Fixed in `ad75f48`. The
  memory-derived state was a median 144 days stale and retrieved analog sets
  overlapping the live ones by a Jaccard of 0.205, calling VOO and CIBR
  bearish while both were bullish.
