# Operator runbook

What to do on your side, what to check, and how to run any job now instead of
waiting for 22:00.

Every command assumes the venv is active:

    cd C:\Users\Oren.Golovchik\Desktop\Oren\invest-by-score
    venv\Scripts\activate

Measured on this machine 2026-10-06. Where something is still broken, it says so.

---

## 1. Your current state, measured

| Thing | State | Action needed |
|---|---|---|
| `NEWS_PROVIDER_API_KEY` | set (32 chars), **quota spent right now** | none — the window reopens on its own |
| `ALERT_EMAIL_USER` / `..._PASSWORD` | set, correct shape | none |
| `ALERT_EMAIL_TO` | not set | none — it falls back to `ALERT_EMAIL_USER` |
| `ALERT_TELEGRAM_TOKEN` | set (46 chars) | none |
| `ALERT_TELEGRAM_CHAT_ID` | **not set** | set it, see §3 — Telegram cannot deliver without it |
| `ALPHAVANTAGE_API_KEY` | not set | optional; without it fundamentals are neutral constants and are deliberately excluded from training |
| `FRED_API_KEY` | not set | optional; macro reports UNAVAILABLE (rebuildable later, FRED serves vintages) |
| Scheduled task | registered, `WakeToRun=True`, ran today 09:18 | none |
| **Both alert channels** | **BLOCKED on this network** | see §2 — this is the one that matters |

All the keys that are set are persisted at **User** level, so the scheduled task
sees them. You do not need to re-set them per shell.

---

## 2. THE THING TO KNOW: alerts cannot reach you on this network

MEASURED today, from this machine:

    api.telegram.org    HTTP 503   (Palo Alto interception, captive portal)
    smtp.gmail.com:587  connects, then TIMES OUT during STARTTLS/login
    newsapi.org         HTTP 200   (works fine)

So **collection works and alerting does not**. The credentials are correct; the
network is eating both channels. This is already documented in
`scripts/verify_telegram.py` as measured on 2026-10-04, and it is unchanged.

What this means in practice: if the collector breaks at 22:00 while you are on
this network, **nothing will tell you**. The code is sound — it is the egress
that is blocked.

**To prove the channels work, run the verify scripts from home wifi or a phone
hotspot:**

    python scripts/verify_alert_email.py          # sends a REAL email
    python scripts/verify_telegram.py             # sends a REAL message

Until one of those succeeds at least once, treat alerting as unproven and check
collection manually (§4). That is the honest position; a credential that is set
but undelivered looks identical to one that works until the first real alert
fires unattended.

---

## 3. Setup you still owe (10 minutes, all optional except the first)

**a. Telegram chat id — the only missing alert-channel piece.**
Telegram is the better channel here precisely because SMTP is blocked, but it
needs the chat id. On a network that allows Telegram:

1. On your phone, open Telegram and send any message to your bot.
2. Then run:

       python scripts/verify_telegram.py --discover

3. It prints the chat id. Persist it so the scheduled task sees it:

       setx ALERT_TELEGRAM_CHAT_ID 123456789

   (`setx` writes it at User level. Open a **new** shell afterwards — `setx`
   does not affect the current one.)

**b. `ALPHAVANTAGE_API_KEY`** — free at alphavantage.co. Without it, every
fundamental metric is a neutral constant (`balance_sheet_quality = 10.0` for
every company), which the training pipeline deliberately refuses to use. With
it, 5 more features become real.

       setx ALPHAVANTAGE_API_KEY your_key_here

**c. `FRED_API_KEY`** — free at fred.stlouisfed.org. Macro currently reports
UNAVAILABLE on every run. Lowest priority: FRED serves historical vintages, so
these days are recoverable later, unlike news.

---

## 4. Daily / weekly checks — where to look

**The one command that answers "is the data flowing?"**

    python scripts/monitor_collection.py --explain-tickers

It prints coverage %, quota spend, failures by kind, missing days, and event
memory freshness — then explains, per ticker, why it has or has not got news.
Today it reads:

    coverage        0.0% (0/75 eligible tickers fetched in the last 7d)
    gaps            22 business day(s) with no news in the last 30d
                    16 of those are past the 7-day provider window and can
                    never be recovered
    event memory    361 record(s) across 26 ticker(s); newest 2026-10-02, 4d old

**What "good" looks like** once the quota recovers: coverage climbing toward
100% over 3 days, `event memory` newest within 1-2 days, `failures` showing at
most `quota_exceeded`.

**What should worry you:**

| Symptom | Meaning |
|---|---|
| coverage stuck at 0% for >2 days | the rotation is not progressing — check the cursor advanced |
| `authentication_failed` in failures | the key is rejected; **this never self-heals** |
| `EXPIRED` on the event-memory line | the newest memory is older than the provider window; no run can extend it from live news |
| `unrecoverable_days` climbing | permanent loss accruing |

**Files to look at directly:**

    data\collect.log                  what the 22:00 task actually printed
    data\collection_report.jsonl      one JSON line per run (the ledger)
    data\collect_cursor.json          rotation position; must MOVE between runs
    data\event_memory.jsonl           the observed/inferred memories themselves

The cursor is the fastest tell. If `collect_cursor.json` has the same `cursor`
value two days running, the rotation is stuck — that was exactly this outage.

---

## 5. Running any job now, without waiting for the cron

**Collection — the perishable one.**

    python scripts/daily_collect.py --dry-run           # costs NO quota
    python scripts/daily_collect.py                     # the real thing
    python scripts/daily_collect.py --sources news      # just news
    python scripts/daily_collect.py --sources prices,fundamentals   # no quota
    python scripts/daily_collect.py --tickers NVDA,AAPL --sources news

Re-running on the same day is **safe**: the ledger is append-only and
`load_raw_records` supersedes older versions by payload hash. A double run costs
bandwidth and changes no answer.

**Check the news key / quota before spending a run:**

    python scripts/verify_news_key.py

Right now this prints `HTTP 429 ... kind=quota_exceeded`, which is how you know
to wait rather than debug.

**Event memory — and the one that needs no quota at all:**

    python scripts/build_event_memory.py --mode forward      # from today's news
    python scripts/build_event_memory.py --mode backfill --dry-run
    python scripts/build_event_memory.py --mode backfill     # from PRICE history

`backfill` is worth knowing about: it generates `inferred` memories from price
history and **spends zero quota**. MEASURED, 3 tickers produced 23 memories.
With 49 of 75 tickers currently having none, this is the cheapest way to give
the forecaster analogs while the news quota is constrained.

**Forecasts:**

    python scripts/run_forecasts.py --dry-run
    python scripts/run_forecasts.py --report        # the scoreboard
    python scripts/run_forecasts.py                # record + close

Today every row comes back `INSUFFICIENT` — 292 of 292 refused. That is not a
bug: `EVENT_MEMORY_MIN_ANALOGS = 5` and only 18 of 75 tickers have that many.
Refusals are recorded on purpose, because the ledger is the denominator.

**Training — works now.**

    python scripts/train.py --ticker NVDA --estimator ridge
    python scripts/train.py --ticker NVDA --all          # all 8 baselines
    python scripts/establish_baseline.py                 # the full suite + trial

Four faults were fixed on 2026-10-06 to get here; see
[news-quota-review-2026-10-06.md](news-quota-review-2026-10-06.md) for what they
were. MEASURED after the fix: `--all` trains 8 estimators over 5 folds, and
`establish_baseline.py` records an incumbent (`historical_mean`, directional
accuracy 0.708) with its trial registered.

Note that a pure baseline winning is a real result, not a failure: it means no
trained model has yet earned promotion.

**Scoring and the dashboard:**

    python main.py NVDA 2026-10-06        # score one ticker
    python web_app.py                     # dashboard at http://127.0.0.1:8000

**Alert channels (sends real messages — run from a network that allows them):**

    python scripts/verify_alert_email.py --dry-run   # checks shape only
    python scripts/verify_alert_email.py             # actually sends
    python scripts/verify_telegram.py --discover     # finds your chat id

**The scheduled task, on demand:**

    Start-ScheduledTask -TaskName "invest-by-score daily collect"
    Get-ScheduledTaskInfo -TaskName "invest-by-score daily collect"

---

## 6. Suggested order, highest value first

1. **Wait for the news quota** (self-clearing) and then run
   `python scripts/verify_news_key.py`. When it says OK, run
   `python scripts/daily_collect.py`. That restarts the signal.
2. **Run the backfill** — `build_event_memory.py --mode backfill`. No quota, and
   it attacks the 49-tickers-with-no-analogs problem directly.
3. **Prove one alert channel from home wifi.** Until then you are flying blind
   at 22:00, and the §4 manual check is the substitute.
4. **Set `ALERT_TELEGRAM_CHAT_ID`** while you are on that network.
5. Add `ALPHAVANTAGE_API_KEY`, then `FRED_API_KEY`.
6. Decide the GitHub Actions question (see
   [news-quota-review-2026-10-06.md](news-quota-review-2026-10-06.md) §6) — and
   if you enable it, set `COLLECT_NEWS_ACTIVE_COLLECTORS = 2` and batch 12, or
   the two collectors will halve each other's quota.
7. Fix the three training faults in §5 as their own piece of work. Nothing in
   the collection path depends on them, so this is not urgent — but the training
   path is currently unrunnable, and that is worth knowing before you rely on a
   model number.
