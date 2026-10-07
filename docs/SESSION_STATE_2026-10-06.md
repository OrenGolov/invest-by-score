# Session state — 2026-10-06

Where things stand, and what the next session should pick up. Read this first.

Branch: `alert-delivery`. Nine commits, listed below in order.

---

## What was wrong when the session started

A daily digest arrived with **77 alerts**, all saying the same thing: that each
ticker "was not looked at today" because of a rotation of "75 per run". Every
clause of that explanation was wrong, and the real state was worse than the
alert described: news capture had been dead for three days.

---

## The commits, and what each one settles

| # | Commit | What it fixed |
|---|---|---|
| 1 | News collection: the deadlock was four defects | Double-fetch, frozen cursor, substring quota detection, wrong alert text |
| 2 | Training: four faults | Embargo gate, union-vs-intersection, stale label version, fold geometry |
| 3 | Fundamentals: a live key bought nothing | Two ratio fields mapped to dollar totals; `valuation_quality` flooring |
| 4 | Scheduling: `-WakeToRun` was missing | Task now wakes the laptop; installer fixed so it stays |
| 5 | Docs | Incident review, runbook, three Word guides |
| 6 | Route fundamentals into training | Flag flipped once the data became real |
| 7 | Backfill: 779 memories, 0 new forecasts | Found the real forecasting blocker |
| 8 | Fundamentals have no axis in a single-ticker dataset | Recorded, not yet verified |
| 9 | This file | — |

---

## The two findings that matter most

**1. Forecasting is blocked by similarity, not by memory count.**

The backfill tripled the event-memory store (361 → 1,140) and moved tickets
meeting the analog minimum from 18 to 67 of 75. Forecasting did not change:
still 292 of 292 refused. The gate is `EVENT_MEMORY_MIN_SIMILARITY = 0.70`:

```
bar    tickers reaching 5 analogs
0.70         0 / 73
0.60        24 / 73
0.50        68 / 73
```

At the shipped bar, **no ticker can produce a conditioned forecast**. Lowering
it is a judgement about how much similarity a published forecast should require,
not a bug fix — registered as open item 29, deliberately undecided.

**2. Fundamentals vary across tickers, not across time.**

The five features now route (NVDA dataset: 17 → 22 columns) and the dataset
still reports all five as zero-variance — because `train.py` builds a
single-ticker dataset whose only axis is time, and one company's fundamentals
are constant down it. They can only contribute in a multi-ticker dataset.
**Not verified**, because Alpha Vantage's free tier is 25 requests/day and the
day's allowance was spent proving the mapping fix. Open item 30.

---

## Pick up here

**First, on a fresh Alpha Vantage quota (resets daily):**

```
python scripts/verify_data_keys.py          # confirm the key serves again
```

Then settle open item 30 — build a multi-ticker dataset and read
`feature_routing["zero_variance"]`. If the five fundamentals drop off that list,
the flag is earning its place. If they do not, turn it off for single-ticker
runs specifically rather than globally.

**When the news quota recovers:**

```
python scripts/verify_news_key.py           # 429 means wait, do not debug
python scripts/daily_collect.py             # then let 22:00 take over
```

Do not raise `COLLECT_NEWS_BATCH_SIZE` to catch up. 25 is sized to leave half
the rolling window spare, and overrunning it is what caused the outage.

**Time-sensitive:** the newest observed event is 2026-10-02 against a 7-day
provider window. News from Oct 2 is still fetchable; in three days it is not.
16 business days are already past recovery.

---

## Still open, in the order I would take them

| Priority | Item | Note |
|---|---|---|
| P1 | Verify item 30 on fresh quota | One command, settles whether fundamentals help |
| P1 | Decide item 29 (similarity bar) | Needs matured outcomes to settle properly |
| P2 | Train the 1d/5d horizons | Less data per conclusion than 20d |
| P2 | Carry ticker/timestamp/regime onto folds | Unblocks two robustness gates (X3/X4) |
| P3 | Wire or delete 4 dead dashboard panels | `confidence_breakdown`, `insights`, `recommended_actions`, `source_reliability` never receive data; `snapshot_hash`, `source_record_ids`, `summary` are sent and ignored |
| P3 | Move 4 ledger-reading tests onto fixtures | They break whenever `establish_baseline.py` runs legitimately |

---

## Operator items waiting on a human

| Item | Blocked by |
|---|---|
| `FRED_API_KEY` | Signup website is blocked on the Aman network — needs home wifi. The API host works at the office, so the key works once saved. |
| `ALERT_TELEGRAM_CHAT_ID` | Same network constraint |
| Prove one alert channel | Telegram returns HTTP 503 and Gmail SMTP times out at the office. **While on that network a 22:00 failure reaches nobody** — `monitor_collection.py --explain-tickers` is the substitute. |
| `.github/workflows/daily-collect.yml` | Committed but not enabled; needs the `NEWS_PROVIDER_API_KEY` secret. If enabled alongside the local task, set `COLLECT_NEWS_ACTIVE_COLLECTORS = 2` and batch 12, or the two halve each other's quota. |

---

## What NOT to do

From `docs/open-decisions.md` section 28, which is the best analysis in this
repo and worth reading before any modelling work:

- **Do not add estimators.** Eight were tested; the chance of a spurious winner
  among eight noise draws is 33.7%.
- **Do not open the sealed holdout.** At 60 rows a fair coin posts 62.7% inside
  the band.
- **Do not report a directional accuracy without its band.** Every number in the
  X1 table is inside the noise.
- **Do not expect accuracy soon.** Detecting a realistic 52-54% edge needs
  385-2,401 *independent* observations; overlapping 20-day windows across 77
  correlated large-cap names means that is 1-3 years of accrual. No modelling
  choice shortens it.

---

## Caveats on this session's own work

- The three `.docx` files were generated with python-docx because node, pandoc
  and LibreOffice are not installed here. **They were never rendered and looked
  at** — structure and content are verified programmatically, page fit is
  estimated.
- `docs/Signup-Guide.docx` is a stale duplicate of `-v2`, left untracked because
  it was open in Word and could not be replaced. Delete it.
- One measurement I ran and discarded: patching
  `EVENT_MEMORY_MIN_SIMILARITY` as a module attribute changes nothing, because
  `find_analogs` binds it as a default argument at import. The first sweep
  measured nothing; the table in item 29 passes it explicitly.


---

# Update — 2026-10-07

## Overnight test result

The 22:00 run did NOT fire on this laptop. Root cause was a power-plan setting
one level below the task: `Allow wake timers` was enabled on AC and **disabled
on DC**, and the machine loses AC power every evening (measured 17:55, 18:33,
17:30 on three consecutive nights — a dock or desk socket, not the charger).
`WakeToRun=True` was necessary but not sufficient.

Fixed: wake timers now enabled on battery. **Untested on this machine** — tonight
is its first real trial.

A separate session on another machine (`sprints-vs`, commit f20160c) proved the
same fix works: three timer wakes, three jobs run, 75 of 77 tickers, 10,771
articles. It also documents a measurement trap I fell into — an S0 Modern
Standby wake logs **no** Power-Troubleshooter event, so judging "did it wake?"
that way is a false negative. Use Kernel-Power 506/507 pairs.

Still unfixed: the task runs as `LogonType: Interactive`, so it needs an active
session. Switching to S4U needs admin (`Access is denied`).

## Alerting

Audited end to end. The path is correct — the task does run the monitor,
credentials are set at User level, a forced CRITICAL verdict renders proper
subject and body. The send still fails, and the diagnostic is right about why:

    tcp smtp.gmail.com:587   connects
    SMTP greeting            NONE (b'')
    https to google          200

That is interception, not a block. No SMTP channel will work here.

**But "no alert can leave this network" is too strong.** The probe that settles
it had never been run:

    api.telegram.org                                 HTTP 503
    gmail.googleapis.com/gmail/v1/users/me/profile   HTTP 401

401 is reachable-but-unauthenticated. A Gmail REST API sender would work from
the office; it needs an OAuth2 refresh token instead of an app password.
Registered as item 31.

## Corrections to the 10-06 session

**Fundamentals routing was reverted.** I enabled it on the strength of today's
snapshot showing three distinct vectors. Training is point-in-time, and 200 of
200 sampled historical cache files are all-null — Alpha Vantage's free tier has
no historical fundamentals endpoint, so past vintages cannot be reconstructed.
Routing them fed five neutral columns to nearly every row. This also settles
item 30: the variance *axis* was never the issue.

## New this session

| Commit | What |
|---|---|
| `7dd72d5` | A quiet ticker is not a failed request |
| `6785924` | Fundamentals need a rotation too (20/run vs a 25/day cap) |
| `087d25f` | Alerting audit: a 443 channel IS reachable |
| `bdbb652` | Revert fundamentals routing |
| `580a201` | Multi-horizon training + a tail bug behind it |

## Twelve-area survey

Most areas are healthier than expected. Three findings worth acting on:

- **Items 32** — `forecast_joint`, `stress_scenarios`, `event_context_panel`
  have NO consumer outside their own gates and tests. `research_view` likewise.
  A gate passing on uncalled code certifies nothing that can affect output.
- **Item 33** — the model registry's three champions are `approved` with
  `has_metrics=False`, while 8 training runs DO carry metrics. There is now a
  measured incumbent to compare against.
- **Sentiment is UNAVAILABLE by design** (no provider exists) and **macro needs
  the FRED key**. Neither is a bug.

## Pick up here

1. Prove one alert channel from home wifi (`verify_alert_email.py`).
2. Check tonight's 22:00 run using **Kernel-Power 506/507**, not the
   Power-Troubleshooter event.
3. Decide items 32 and 33 — both are "wire it or delete it" calls.
