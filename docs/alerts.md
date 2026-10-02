# Alerts: the monitoring system

Every finding reaches you in two places — the **Monitoring** tab of the local
website, and your **email** — and nothing is ever deleted.

This document is the operator's guide: what to set up, what to expect, and how to
read what arrives.

---

## Setup, once

### 1. Email credentials

Email needs a Gmail **App Password**, not your account password. Google rejects
the account password for SMTP, and the error it returns is unhelpful, so the
sender names this case explicitly if it happens.

Create one at <https://myaccount.google.com/apppasswords> (it requires 2-step
verification on the account), then set two variables at **User** scope:

```powershell
[Environment]::SetEnvironmentVariable("ALERT_EMAIL_USER","you@gmail.com","User")
[Environment]::SetEnvironmentVariable("ALERT_EMAIL_PASSWORD","<16-char app password>","User")
```

Optionally send to a different address:

```powershell
[Environment]::SetEnvironmentVariable("ALERT_EMAIL_TO","someone-else@example.com","User")
```

Then **sign out and back in**, or reboot. A variable set only inside a terminal
session is invisible to Task Scheduler — that is precisely how five nights of news
were lost before this was understood.

**Nothing breaks without these.** Every alert is still detected, graded and stored,
and the Monitoring tab shows all of them. Only the email channel is disabled, and
the run reports that it is.

### 2. Schedule the tasks

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_daily_task.ps1
```

Three tasks, split by what is recoverable:

| time | task | why then |
|---|---|---|
| 14:30 | news + event collection | **Perishable.** News is gone after 7 days, and the NewsAPI free tier's 100 requests/day are largely spent by evening. |
| 22:00 | prices, fundamentals, macro | After the US close. All re-fetchable, so a missed run costs nothing. |
| 22:15 | alert detection and delivery | After collection, so the detectors see the day's data. |

Verify with `Get-ScheduledTask -TaskName "invest-by-score*"`.

---

## The five priorities

Priority is **derived**, not asserted. The detectors report a two-value severity
(`info`/`warn`); that answers *how bad*. What answers *how soon* is the kind of
change and whether it has been confirmed:

| priority | what earns it | email |
|---|---|---|
| 🔴 **Urgent** | A measurement that existed yesterday cannot be made today. The evidence base itself changed. | immediately |
| 🟠 **Very High** | A confirmed adverse change, or a high-impact event type. | immediately |
| 🟡 **High** | A confirmed change that is not adverse, or a measurement becoming available. | immediately |
| 🔵 **Medium** | A change that has not yet persisted, or an event whose type has too little history to size. | daily digest |
| ⚪ **Low** | An event type whose realized history shows a small move. | daily digest |
| ⚫ *(ungraded)* | **A detector could not reach a verdict.** | daily digest |

The last row matters most. An ungraded alert is **not** a quiet one — it means
something is blind, which is the finding a monitoring system must never swallow.
It carries no priority, because assigning one would claim evidence the detector
explicitly refused to give, and it is never dropped.

### Why Urgent is a pipeline failure, not a market crash

This surprises people. A1 measured that folding an availability change into a
magnitude turns a vanished forecast into a −0.56 crash that never happened. So a
measurement *disappearing* is reported as the most urgent thing the system can
say — because the system has stopped being able to see, and you need to know today.

---

## The recommended action

Five actions exist in the vocabulary: Buy, Sell, Hold, Watch, Review. **Only
`Watch` and `Review` are ever emitted.** This is a measured constraint, not
caution:

- X10's release gate reports the current release **NOT APPROVED**, with 2 of 9
  gates passing, and rule 5 disables real-capital execution until they pass.
- X1 measured four of five estimators scoring *below a coin flip* (0.400–0.427
  accuracy); only `historical_mean` beat chance, at 0.720.

A `Buy` emitted from an alert would be a directional recommendation from a system
that has measured itself unable to make one. `Watch` means "a real graded change,
keep looking at it". `Review` means "something about the evidence needs a person".

A test asserts that no state in any detector vocabulary can produce Buy or Sell.

---

## How many emails to expect

Roughly **a handful per weekday**, concentrated in the digest.

The measurements behind that: A4 found regime alerts run 7.6 per ticker-year at
3-session confirmation, which across 77 tickers is ~585/year. A7 found that
**93.3% of alert-days are repeats**, and suppression removes them — an unsuppressed
channel emits 15.01 alerts per episode, and nobody reads the fifteenth.

Duplicate prevention works because an alert's identity **excludes every
timestamp**. The same unchanged finding observed on two days shares one ID and is
emailed once. A state *change* produces a new ID and a new email. This is the same
trap measured twice before in this codebase: a key containing `as_of` is unique
every day and suppresses nothing.

A **failed** send never marks an alert as delivered, so an SMTP outage delays mail
rather than silencing it permanently.

---

## Reading an email

```
Subject:  Very High: AVGO - Earnings Results

Action: Watch

🔴 IMPORTANT - Immediate Review Required
Relevant from: 2026-10-01
🟢 Active Monitoring Date: 2026-10-29
Evaluation horizon: 20d

Who:   Broadcom (AVGO) — reported by Reuters
What:  Earnings Results: Broadcom reports quarterly earnings above estimates
When:  Published 2026-10-01T20:30:00+00:00; assessed for 2026-10-01.

Why it Matters:
  - This event type's median absolute 20d move is 5.48% across 1954 remembered outcomes.
  - The 90th percentile move is 20.68%.
  - The article reads positive (tone +1.00), which indicates direction rather than size.
```

Two things to notice:

**The "Why" quotes measurements, never adjectives.** "5.48% across 1,954
remembered outcomes" is a number you can check. The alert does not say the news is
important; it says how far this *kind* of news has historically moved the price.

**Tone is direction, not size.** A cheerful press release and a grim one about the
same event type grade identically, because they move the price by the same measured
amount. Tone is reported and deliberately excluded from the grading — otherwise a
glowing announcement would outrank a quiet regulatory filing.

---

## The Monitoring tab

Open the site and click **Monitoring**:

```powershell
python web_app.py
```

Then <http://127.0.0.1:8000>.

- **Newest first**, always.
- **Collapsed cards** show priority, ticker, title, time. Click to expand the full
  detail, including *why* that priority and *why* that action — so the card carries
  the evidence for its own grading and you never have to re-run anything.
- **Filters**: priority (click the coloured chips), ticker, event type, date range.
- **Search** covers ticker, title, state, reason, priority and action.
- **Times are shown in your timezone**, converted from UTC. Storing local time
  would be unsortable across a daylight-saving boundary; a date filter sends your
  browser's offset so "today" means *your* today. That was a real defect: before it
  was fixed, filtering today returned 0 of 12 rows on the day the feed was
  populated.
- **Nothing is ever deleted.** Widening the date range always shows the full
  history.

Works at phone width.

---

## Running it by hand

```powershell
python scripts\run_alerts.py                      # detect, store, deliver
python scripts\run_alerts.py --dry-run            # detect and render, send nothing
python scripts\run_alerts.py --no-email           # store only
python scripts\run_alerts.py --tickers NVDA,MSFT
python scripts\run_alerts.py --digest-only        # just send today's digest
python scripts\run_alerts.py --detectors regime   # one detector
```

Running it twice is safe. Re-detection appends to the ledger, and the ID-based
dedup means nothing is emailed twice.

---

## What the detectors can and cannot see

| detector | what it finds | state today | cost per ticker |
|---|---|---|---|
| `regime_change` | A4 — regime flips, confirmed over 3 sessions | **nightly** | 0 requests |
| `news_event` | A9 — corporate news, graded by its type's realized history | **nightly** | 0 requests |
| `thesis_break` | A5 — evidence turning against the case | **working, opt-in** | 1 news request |
| `forecast_change` | A1 — the forecast itself moving | needs a trained model | — |
| `confidence_change` | A2 — how much to trust it, and *why* it is uncertain | needs a trained model | — |
| `event_impact` | A3 — high-impact events from stored outcomes | used *by* A9 | — |
| `forecast_threshold` | A6 — all three conditions crossing, no veto | **cannot fire**: zero of twelve condition inputs exist | — |

### Why `thesis_break` is opt-in

It works. Measured on a live MSFT decision the attribution is real — operational
**+6.41** (supports), narrative and macro neutral, carrier `operational` — and a
second run correctly reads the prior from the ledger and reports `NONE` for an
unchanged thesis.

But it reaches that attribution through `orchestrate_score`, which **fetches
news**: one request per ticker. Across 77 tickers against a 100/day tier that is
the quota defect this runner already had once. So the nightly sweep runs `regime`
and `news` only — both cost zero requests — and you run the thesis detector
deliberately:

```powershell
python scripts
un_alerts.py --detectors thesis --tickers MSFT,NVDA,AVGO
```

A handful of tickers at a time is affordable. The whole portfolio is not, until
the news budget question (`docs/open-decisions.md` item 7) is settled.

A6 is honest about being dead: it reports `NOT_EVALUATED` rather than silence,
because inferring calm from an unevaluable condition is how a live alert goes
missing.

### The news grading limit, measured

Of 2,921 stored event memories, only `earnings` carries a 20-day response — 1,954
usable outcomes, median absolute move 5.48%. Every other type has 1-day responses
only:

```
earnings                1,969 memories   1,954 usable at 20d   median 5.48%
other                     776                0
product_launch             42                0
litigation                 37                0
management_commentary      31                0
m_and_a                    28                0
regulation                 19                0
macro_shock                16                0
guidance                     3                0
```

So at the 20-day horizon exactly one event type can be sized, and the rest
correctly report `UNKNOWN_IMPACT` → Medium. That is the honest answer, not a gap:
a median estimate at n=3 spans a five-fold range across resamples. It improves as
outcomes close.

---

## A note on which articles count

Fifteen of your holdings have a ticker that is an English word or a colliding
abbreviation — `V`, `BE`, `NOW`, `CAT`, `KO`, `ARM`, `MP`, `KEEL`, `STX`, `TER`,
`NU`, `GLW`, `AEP`, `ANET`, `CEG`.

Measured over 6,360 captured articles, matching on the bare symbol admitted 116
articles for those tickers and **93 were not about the company at all** — a
Chevrolet Corvette for ARM, cat memes for CAT, a boxing match for KO, a PyPI
package for KEEL. For `V`, all ten were collisions.

Those fifteen now require the **company name** or **finance language** alongside
the symbol. The effect:

| | before | after |
|---|---|---|
| articles admitted on those 15 | 116 | **10** |
| articles admitted on the other 62 | 1,633 | **1,633** |

Nothing changed for `NVDA`, `MSFT` or any other symbol that does not collide with
ordinary English.

If you add a holding whose ticker is a common word, add it to
`NEWS_COLLISION_PRONE_TICKERS` in `core/config/_base.py` with the company name. A
config validator refuses an entry with no name, because such a ticker would reject
every article about it and go permanently dark.

---

## When nothing arrives

Check, in order:

1. **Was anything detected?** `python scripts\run_alerts.py --dry-run --no-email`
2. **Did the task run?** `Get-ScheduledTaskInfo -TaskName "invest-by-score alerts"`,
   and read `data\alerts.log`.
3. **Are the email variables visible to the scheduler?**
   `[Environment]::GetEnvironmentVariable("ALERT_EMAIL_USER","User")` — if this is
   empty, the variable was set in a terminal only.
4. **Were they suppressed?** `data\alert_deliveries.jsonl` records every attempt,
   including `SKIPPED` and `FAILED` with the reason. A suppressed alert is still in
   `data\alerts.jsonl` and still on the dashboard.

Silence from this system should always be explainable from those four files. If it
is not, that is a defect worth reporting.
