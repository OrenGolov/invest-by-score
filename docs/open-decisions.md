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

### RE-MEASURED 2026-10-01 in E4 — unchanged, and one fact worth adding

```
universe entry    ticker + reason "initial_member", no type, no listed_from
price history     119 bars, 2026-03-31 -> 2026-09-18   (210 required)
```

**It is a 2026 listing**, which the original note did not say. The history
shortfall is therefore self-resolving: at ~252 sessions a year it crosses 210
bars around March 2027 without anyone doing anything. Only the METADATA question
needs a decision — what the instrument is — and until that is answered the news
skip is correct regardless of history length.

## 3. Breadth / participation features — PARKED

**Raised:** Sprint C. **Blocked on:** an index-constituent adapter.

Explicitly deferred with a `breadth` entry in `fetch_data.SOURCE_REGISTRY`
(`provider_key_required`, `base_confidence: 0.0`, `domain: market_breadth`).

**Why it cannot simply be added:** survivorship-safe breadth requires knowing
which names were *in* the index on each past date. It can never be inferred
from price, volume or technical indicators — inferring it from today's
constituents is survivorship bias by construction. Held open by 18 tests in
`tests/test_deferred_features.py`, including the placeholder itself.

### RE-CHECKED 2026-10-01 in E3 — the blocker is real and unchanged, STAYS PARKED

The one thing that could have unblocked this is the V6 universe ledger, which
carries `listed_from` / `listed_to` fields that look like point-in-time index
membership. They are not:

```
data/universe.jsonl      77 rows
listed_from populated     0
listed_to   populated     0
source_id                 portfolio_list_snapshot (all 77)
```

It is a **portfolio watchlist**, not index membership. Breadth asks "how many of
the S&P 500 are above their 200-day average" — that needs the index's
constituents on each past date, which nothing in the repository has and no
provider is connected for.

**Revisit when** an index-constituent provider is configured. Until then the
deferral is the correct answer and the placeholder's `base_confidence: 0.0` is
doing its job: a breadth feature that could not be computed must never reach a
model as a neutral value.

## 4. The source → outcome join — BUILT in B2 (was PARKED)

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

### BUILT 2026-09-29 in B2 — and the parked measurement was right while its conclusion was wrong

`core/source_outcome_join.py`. The article-level `ticker` field is indeed `None`
for every stored article, exactly as measured. But **every raw envelope carries a
`request_key` of the form `MSFT_2026-09-17`**, naming the ticker the fetch was
made for:

```
articles total                3,753
with an article-level ticker      0   <- the parked measurement, correct
with a request_key ticker     3,753   <- and every one is attributable
distinct outlets                 58
```

So no re-ingestion was needed. **The actual missing link was the CLASSIFICATION,
not the ticker:** `tone` is `None` for all 3,753 articles, so the first run of the
join dropped every one of them. `news_adapter.resolve_tone` derives a tone from
the headline, and the derivation is now stamped on each observation.

**MEASURED RESULT, at the 1d horizon:**

```
1,026 observations across 13 outlets   (join rate 27.3%)
L4 accepted 1,026 of 1,026
per-outlet hit rates 0.256 -> 0.836
```

L4 is learning for the first time. At 5d the join yields 48 observations and at
20d none — correctly, because the news spans Aug 24–Sep 21 while prices end
Sep 18, so a longer horizon has not matured. That is PIT behaviour, not a defect.

**Two weaknesses recorded on every observation rather than hidden:**
1. `attribution` is `request_key`, not `entity_resolution` — the ticker is
   inferred from the fetch, which is weaker than resolving it from the text.
2. `tone_derivation` names the v1 lexicon, which **agreed with the obvious
   reading on only 5 of 6 hand probes** (it scored "Stocks Settle Sharply Higher
   as Bond Yields Fall" as −1.00, catching "Fall" and missing "Higher"). An
   outlet's measured hit rate therefore partly measures the lexicon.

**What remains open:** weakness 2 is the one worth closing, and it is a
classifier problem rather than a join problem. A hit rate built on a lexicon that
is wrong one time in six has a noise floor no amount of extra data will lower.

## 5. Cluster exposure — BUILT in B4 (was PARKED)

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

### BUILT 2026-09-29 in B4 — and the definition was chosen by measurement

`core/cluster_exposure.py`. **Correlation clustering**, because it catches what
sector labels cannot: SOXX is an ETF, and the correlation binding these four
names is not a label. Both linkage rules were swept across 13 thresholds on the
real book before choosing:

```
thr    single   average
0.450   100.0%    84.5%
0.500   100.0%    52.6%   <- average finds exactly the four semis
0.550   100.0%    36.3%
0.600    52.6%    36.3%   <- single finds them only here
0.650    36.3%    36.3%
```

**A first reading of this was wrong.** I took average linkage to be the more
*stable* rule; the swept spreads are nearly identical (63.7% vs 62.3%). The real
discriminator is **degeneracy**: single linkage merges on one qualifying pair, and
SOXX correlates 0.55–0.76 with everything while MSFT–GOOGL is 0.59, so it chains
the semis to the non-semis and reports the **whole book as one cluster at 5 of 13
thresholds**. A cluster containing every holding cannot distinguish a concentrated
book from a diversified one. Average linkage degenerated at 0 of 13.

**Shipped result, recomputed by the gate from tracked price frames:**

```
worst single position       16.3%   (R1 level 40%) -> no flag
the four semis TOGETHER     52.6% of variance on 40.0% of weight
mean pairwise correlation    0.62
verdict                     CONCENTRATED
```

**Three defects the sabotage test found, all fixed:** a ragged covariance raised
`IndexError` and *crashed* instead of reporting `NOT_EVALUATED` — an unreachable
fail-closed path, which is worse than an untested one because it looks like a
guarantee; and two gaps where only one of the gate/tests pair caught a mutation.
The synthetic test fixture also needed a deliberate *bridge* asset before it could
exhibit degeneracy at all — absent one, the two linkage rules are genuinely
equivalent, and a test sweeping a bridgeless fixture would have concluded exactly
that.

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

**RECURRED 2026-09-28, inverted, in X4.** `check_event_robustness.py` and
`test_there_are_no_event_memories` asserted `len(load_memories()) == 0`. That
is TRUE on a fresh clone and FALSE here (2084 rows), so this pair **passed in
CI and failed locally** — the exact mirror of the original. Practice item 2
alone would never have caught it: a clean-clone run reports green.

So the rule is stronger than "verify against a clean clone". **A check keyed to
gitignored data is wrong in both directions, and neither environment alone
reveals it.** The fix was not to tolerate the absence but to find the invariant
that does not depend on the untracked store: X4's blocker is that no training
fold carries per-observation event attribution, measured from the TRACKED
`data/training_runs.jsonl`. The memory count is now reported as context and
decides nothing.

4. A check must assert an invariant of the REPOSITORY. If the deciding
   measurement reads a gitignored path, the check is measuring the checkout —
   test it with that data both PRESENT and ABSENT before trusting it.

**RECURRED A THIRD TIME 2026-09-28, in X10** — and this one had nothing to do
with gitignored data, which is what makes it the useful instance.
`check_honest_gate.py` asserted the gate counts measured in my working tree:

    working tree (edited)   X9 DIRTY   -> 0 gates pass, 3 explicit non-passes
    fresh clone (clean)     X9 FROZEN  -> 1 gate passes,  2 explicit non-passes

X9 is the one gate whose verdict depends on the checkout rather than the model —
that is exactly what it was built to detect — so any count including it asserts a
property of the working directory. The clean-clone run caught it before CI did,
one commit after it was pushed.

**So the rule generalises past its original cause.** The pattern is not
"gitignored data"; it is ASSERTING ANYTHING THAT VARIES BETWEEN CHECKOUTS —
untracked files, a dirty tree, local caches, machine state. Three instances in
two days (`9e7852d`, `2aa15f2`, `755e94b`), each with a different surface and one
root:

5. Before asserting a count or a verdict, ask WHAT WOULD CHANGE THIS NUMBER ON
   ANOTHER MACHINE. If the answer is anything other than the tracked contents of
   the repository, assert the checkout-independent part and handle the rest
   separately. X10 now asserts the eight EVIDENCE gates (which read only tracked
   ledgers) and checks X9 only against its two legitimate states.

**RECURRED A FOURTH TIME 2026-09-30, in D2.** `check_type_coverage.py` failed
when the mypy error count dropped more than 5 below its baseline — a guard I added
so a stale baseline could not accumulate slack. CI #164 failed on it while every
local run passed, including one against a fresh clone.

**The count is PLATFORM-dependent, not checkout-dependent**, which is why the
clean-clone check did not catch it: the 83 was measured on Windows/Python 3.13
and CI runs Ubuntu/Python 3.12, where mypy resolves platform-specific stubs
differently. A baseline measured on one machine cannot bind another.

**And the failure mode was backwards.** The gate broke the build for the code
being CLEANER than recorded. Progress must never fail a build.

6. A ratchet asserts a DIRECTION, not a value. It may fail on a regression and
   must only REPORT an improvement — and its threshold must not depend on the
   interpreter, the OS or the tool version, none of which the repository pins for
   a developer's machine.

## 7. Collection cadence — MEASURED in E1/E2, still an operator decision

**Raised:** Sprint L (2026-09). **Owner:** operator.

The daily collector and the forecast runner are both correct to schedule now
(the live chart-state blocker closed in `ad75f48`), but nothing is scheduled:
the operator asked to run "every other day if needed, but not now".

**What would settle it:** whether to cron them, and at what cadence. The
NewsAPI free tier's 100 requests/day is the binding constraint, which is why
`COLLECT_NEWS_TRACK_ANYWAY` exists for SOXX and CIBR.

### MEASURED 2026-10-02 — THE 22:00 SLOT IS THE WORST POSSIBLE TIME

A later finding that changes the recommendation, and it is a TIMING defect rather
than a quota one. Reading `data/collection_report.jsonl` across every scheduled
night:

```
as_of        overall   news      ok  unavailable  note
2026-09-21   FAILED    FAILED     0            1  quota_exhausted on ticker 1
2026-09-28   FAILED    FAILED     0            1  quota_exhausted on ticker 1
2026-09-29   FAILED    FAILED     0            1  quota_exhausted on ticker 1
2026-09-30   FAILED    FAILED     0            1  quota_exhausted on ticker 1
2026-10-01   FAILED    FAILED     0            1  quota_exhausted on ticker 1
2026-10-02   PARTIAL   PARTIAL   33            7  ran mid-afternoon instead
```

Every scheduled night failed on its FIRST ticker in 0.2-0.3 seconds with
`quota_exhausted: true`. A probe of the provider confirms the key is valid and the
quota is available during the day (HTTP 200, 242 results), so this was never a
missing or broken credential — **the day's 100 requests were already spent by
22:00**.

Where they go:

```
one collector run                40 requests
a second collector run           40 requests   -> 80 of 100
every /api/score call             1 request    per ticker scored
```

So an afternoon of ad-hoc scoring consumes the remaining 20 and then eats into the
budget the evening run needs. Scheduling news collection at 22:00 puts the
PERISHABLE source last in the queue for a resource that depletes all day — the
exact inversion of what irreversibility demands. Price bars, fundamentals and
macro are all re-fetchable; news is gone after `NEWS_LOOKBACK_DAYS = 7`.

**Acted on rather than left open**, because this one is not a preference: the
collector now runs at 14:30 local, before the US close but ahead of the day's
ad-hoc consumption. The 22:00 slot remains for the non-perishable sources. The
operator still owns the one-run-vs-two question below, and the free-tier-vs-upgrade
question in item 8.

### MEASURED 2026-10-01 in E1/E2

**The arithmetic of the quota, which decides the cadence:**

```
eligible tickers              75
news batch per run            40   (COLLECT_NEWS_BATCH_SIZE)
free tier limit          100/day
```

So a full sweep needs **two runs per day**, and the rotation cursor exists
precisely because one run cannot cover the universe. Today's run covered 40 of 75
before the provider returned HTTP 429 — meaning the day's quota was already partly
spent.

**The collector behaved correctly under the limit**, which is worth recording
because it is the fail-closed contract working on live data:
- it stopped early rather than retrying into a hard limit;
- it did **not** advance the rotation cursor, so the next run retries the same
  tickers rather than skipping them;
- it named the perishable loss explicitly — *"news — this day cannot be recovered
  later"*.

**The key is valid.** `scripts/verify_news_key.py` confirms it is set and
well-formed; the 429 is the free tier's daily limit, not a configuration fault.

**What is genuinely lost:** 2026-09-22 to 09-25. A price bar can be refetched from
history; a news window closes. Those four days have no news and never will.

**What is captured:** Sept 28–30 and Oct 1 are now committed (price and
fundamentals). Storage runs at ~1.65 GB/year at this cadence — below the 3.9
GB/year that made the news ledger local in `0e289b4`, but the same order of
magnitude, so the same decision will arrive again.

**The decision remains the operator's**, and it is now two questions rather than
one:
1. Register the scheduled task (`scripts/install_daily_task.ps1`) — and at **twice
   daily**, not once, or the universe is never fully swept.
2. Accept the free tier (≈1.3 sweeps/day of headroom) or upgrade. At the free tier
   the news evidence will always be partial, which caps what L4 and X4 can ever
   measure.

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

### CONFIRMED 2026-10-01 in E4 — it happened three more times, and stays OPEN

The improvement sprints produced three further instances, so the pattern is now
nine occurrences across two sprint families:

| task | sabotage that wrongly passed | why the probe could not decide |
|---|---|---|
| X9 | the `if absent:` branch disabled | gate and tests both drove verdicts through a LOCAL helper that duplicated the rules, never the shipped builder |
| X9 | the unprobed-component fallback disabled | every declared component already had a probe, so the branch was unreachable |
| B4 | the NOT_EVALUATED verdict swapped | a ragged covariance raised `IndexError`, which was not in the except clause — the branch CRASHED instead of running |

**The B4 one is the sharpest version of the pattern yet.** The sabotage passed
because the fail-closed path it targeted was *unreachable*, and an unreachable
fail-closed path is worse than an untested one: it looks like a guarantee and is
not.

**A second habit this session added**, beyond checking reachability: when a
sabotage passes, check whether the TEST FIXTURE can express the property at all.
B4's synthetic covariance never degenerated at any threshold, so a sweep over it
would have concluded the two linkage rules were equivalent — which, absent a
broadly-correlated bridge asset, they genuinely are. The fixture needed the
bridge before the test could mean anything.

**What would settle it:** a convention that every sabotage is accompanied by a
positive control — a paired scenario that FAILS when the guard is removed and
PASSES when restored — so an inert probe is visible rather than silent. A7 does
this ad hoc (`unchanged_old` beside `relieved`); making it a rule would cost
little and remove the ambiguity.

---

## 10. F7's structured assessment — ACCESSOR EXPORTED in B3 (was OPEN)

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

### SETTLED 2026-09-29 in B3 — the first option

F7 now exports `confidence_of`, `band_of`, `binding_factor_of` and
`summarise_assessment`. Both existing consumers were converted to them, so the
reading convention is shared rather than re-derived:

* **A6** (`confidence_alert._read`) hand-rolled the field names and the `[0, 1]`
  check; it now calls the accessors and translates F7's error into its own, so a
  caller catching `ConfidenceAlertError` still sees a malformed assessment.
* **D1** (`research_view._scalar`) is a GENERIC table-cell renderer, so it was
  not converted wholesale — that would couple the renderer to F7. It now detects
  an F7 assessment by F7's own `assessed_object` marker (not by shape, since
  guessing is the defect) and reads through the accessor, falling back to the
  generic path for any other mapping. A malformed assessment renders
  `{invalid}` rather than crashing a view whose job is to show state.

`summarise_assessment` is the short form D1 lacked: under 300 characters against
the ~4,000-character cell that started this, counting the measured and
unmeasurable factors instead of listing them.

**The trap is pinned by a test.** The assessment carries `binding_value` (0.42)
and `weighted_sum` (0.91) beside `confidence` (0.83), and the test asserts the
accessor returns none of the wrong ones — so a future "simplification" to "take
the first number" fails rather than silently thresholding on a factor score.

---

## 11. `scripts/train.py` could not regenerate its own ledger — FIXED in A2

**Raised:** Sprint X8 (2026-09-28). **Found by:** the sealed-holdout gate,
while trying to establish where the holdout fell.

**MEASURED.** F2 (`34bd464`, 2026-09-19) set the longest label horizon to 252
sessions. `build_walk_forward_folds` requires `embargo >= max label horizon`.
`data/training_runs.jsonl` was written 2026-09-18, under a max horizon of 60,
and `scripts/train.py` still ships `--embargo-sessions 60` as its default. Both
are now rejected:

    ledger geometry (406 rows, fold=120, embargo=60, holdout=60)   REJECTED
    scripts/train.py defaults (fold=80, embargo=60, holdout=60)    REJECTED

**And no legal geometry fits the data.** With the embargo pinned at 252, the
cheapest walk-forward layout that still reserves a holdout needs 432 rows
against the 406 the dataset has — short by 26. The current default geometry
(252/252/126) needs 882, short by 476.

**Why it matters.** The 8 shipped baselines cannot be retrained by the script
that produced them, so every number they carry is unreproducible in the strict
sense X9 checks. X8 reports this honestly (`NOT_EVALUATED`, shortfall named)
rather than papering over it, but the underlying condition stands.

**What would settle it:** one of three, and the choice is a modelling decision
rather than a bug fix —
1. extend the dataset past 432 rows (26 more prediction times at the cheapest
   legal geometry, ~500 for comfort), then retrain and reseal;
2. drop the 252-session horizon from `LABEL_HORIZON_SESSIONS` if a one-year
   label is not actually wanted, which returns the embargo floor to 120;
3. keep both and accept that the 252d horizon is not trainable on this dataset,
   documenting which horizons a run may legitimately claim.

Option 1 is the only one that preserves what F2 set out to do. Until then
`scripts/train.py --embargo-sessions 60` is a default that cannot run, which is
worth fixing on its own regardless of which option is chosen.

### FIXED 2026-09-30 in A2 — option 1, and the defaults are now derived

```
scripts/train.py defaults   rows=1464  fold=300  embargo=252  holdout=60
build_walk_forward_folds    2 folds, holdout [1404, 1463]      ACCEPTED
```

The defaults are **derived from `LABEL_HORIZON_SESSIONS`** rather than hardcoded,
so the next horizon change cannot strand them the way F2 did. 1,464 is measured,
not chosen: it is the smallest row count yielding two walk-forward folds *and* a
holdout embargoed against the 252-session horizon.

The dataset was never data-limited — 3,772 sessions of 15-year history exist per
ticker against the 406 rows the old ledger was built from. That 406 came from
`--tickers 12 --times 25` build parameters, not from a shortage of data.

**A2 also found a label-leakage hazard underneath this**: the geometry's embargo
is counted in ROWS while the horizon is in SESSIONS, so a 14-ticker panel's
252-row embargo spans only 18 dates. `train_baseline` now verifies the calendar
separation before fitting anything.

---

## 12. Symbol collisions in entity resolution — FIXED 2026-10-02

**Raised and fixed:** 2026-10-02, while wiring A9.

`news_adapter.resolve_relevance` scored 1.0 whenever a ticker appeared "as a token
in the text". MEASURED over 6,360 captured articles, that admitted **116 articles
on the 15 holdings whose symbol is an English word or a colliding abbreviation, of
which 93 were not about the company at all**:

```
ARM    35-Years-Owned 1972 Chevrolet Corvette Coupe Project
CAT    27 Meowing Memes Ministering Mood Boosts for Cat People Like You
KO     Tyson Fury vs Anthony Joshua ... smiles then KO
MP     Proposed deal to resolve Drumcree dispute, says DUP MP
KEEL   keel-workflow 1.25.0
NOW    Surface Pro, 13-Inch (11th Edition) $1,747 @ Microsoft Store
V      10 of 10 admitted articles were collisions
```

### I ARGUED AGAINST FIXING IT ON A BLAST RADIUS I HAD NOT CHECKED

The first version of this entry said `resolve_relevance` is "shared with the
scoring engine, N1 entity resolution and the memory builder", and used that as the
reason to record rather than fix. **That was wrong.** Neither `score_engine.py`
nor `event_memory.py` references it. There are two production call sites, both in
the news path, plus `event_contract` reading the resulting *field*. The fix was
tractable from the start, and the deferral was based on an assumption rather than
a grep.

### THE RULE, AND WHY IT IS THE RIGHT ONE

For the 15 listed symbols only, a bare-token match must be CORROBORATED by the
company name or a finance marker:

```
collision-prone tickers   116 -> 10 admitted
the other 62 holdings    1633 -> 1633 (untouched)
```

The separation is what justifies it. AEP and CEG kept **100%** of their articles —
every one was genuinely financial — while V, NOW, STX and BE kept **0%**, because
none ever were. A rule that cut indiscriminately would have dropped the first
group too.

Verified both ways: **15 of 15** genuine articles admitted, **15 of 15** measured
collisions rejected.

### THE FIX WENT WRONG IN BOTH DIRECTIONS IN TURN

Recorded because it is the useful part:

- **Too permissive.** An unbounded marker match let `rally` match inside
  *lite**rally*** and *neut**rally***, readmitting a baseball report for ARM.
- **Too strict.** The boundary fix for that was written through a shell heredoc,
  which turned its `\b` anchors into literal **backspace characters**. The
  pattern then matched nothing, rejecting genuine coverage *and invalidating the
  measurement taken against it* — with no test failing. Markers are now
  precompiled once, in one visible place.

A third defect surfaced while verifying: the company-name branch reads
`record["company_name"]`, which MEASURED on live NewsAPI records is always `None`.
So "Caterpillar raises full-year guidance" — no bare symbol, no provider ticker —
had no path to admission at all once the symbol rule tightened.

**A list, not a heuristic.** "Is this symbol an English word?" needs a dictionary
and a judgement per word (is ARM a word? BE? MP?). The list is explicit and
auditable, and each entry carries the name that corroborates it, which a heuristic
would have needed anyway.

**What is NOT changed:** the three-valued scale, the 0.7 floor, the 62 other
holdings, and a provider-asserted ticker match — the provider resolved the entity
itself, which is a stronger claim than the symbol appearing in text.

### RE-SCORING THE STORED MEMORIES — ANSWERED 2026-10-03: NOT NEEDED

I had recorded this as still open, on the assumption that memories built under the
old rule "carry collision-sourced events". **MEASURED, none of them do.**

```
memories in the store                              780
referencing a news source record                     0
inference_method                quarterly_volume_cadence (all 780)
entity_resolution_method                      "" (all 780)
on collision-prone tickers                         145
```

The collision bug lived in `resolve_relevance`, which decides whether a NEWS
ARTICLE is about a ticker. Every stored memory is `provenance: inferred` — dated
from price behaviour by the quarterly volume cadence, with no article involved at
any point. The 145 memories on collision-prone tickers are price-derived too, so
the ticker is the one the backfill ASKED FOR, never one matched out of article
text. There is nothing for the fix to re-score.

This only becomes a live question once `observed` memories accrue — those come
from news and would have passed through `resolve_relevance`. None exist yet,
because the forward mode writes only what the provider reports and the 7-day news
window has never been captured with a working key on a schedule. From here on they
will be written under the corrected rule.

**A SEPARATE LOSS, RECORDED HONESTLY.** The store held 2,921 memories; it now holds
780. I deleted `data/event_memory.jsonl` with an `rm -f` while resetting alert test
data on 2026-10-02 — the second gitignored store I destroyed that way — and it was
not recoverable from git. The backfill rebuilt the earnings memories from price
history, which is why the count is lower: the ~2,140 lost were non-earnings types
(litigation, product_launch, management_commentary and the rest) carrying 1d
responses only, which were never gradeable at the 20d horizon. Grading is
unaffected — 780 analogs against a floor of 5 — though the measured median 20d
move moved 0.0548 → 0.0690. `scripts/check_irreplaceable_stores.py` now reports
such a loss instead of letting it hide.

## Closed

- ~~**Live runs must use today's chart state.**~~ Fixed in `ad75f48`. The
  memory-derived state was a median 144 days stale and retrieved analog sets
  overlapping the live ones by a Jaccard of 0.205, calling VOO and CIBR
  bearish while both were bullish.
