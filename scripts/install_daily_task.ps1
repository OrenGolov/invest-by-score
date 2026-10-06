# Register the daily tasks with Windows Task Scheduler.
#
# THREE tasks, because they have three different timing constraints:
#
#   1. news collect   14:30  the PERISHABLE source, scheduled EARLY (see below)
#   2. daily collect   22:00  prices, fundamentals, macro -- all re-fetchable
#   3. alerts          22:15  detect, store and email, after collection
#
# WHY NEWS RUNS AT 14:30 AND NOT AT 22:00. MEASURED across every scheduled night
# from 2026-09-21 to 2026-10-01, news failed on its FIRST ticker in 0.2 seconds
# with `quota_exhausted: true` -- five consecutive nights, permanently lost. The
# key was valid the whole time; a daytime probe returned HTTP 200. The day's 100
# NewsAPI requests were simply already spent by 22:00:
#
#       one collector run        40 requests
#       a second run             40 requests  -> 80 of 100
#       every /api/score call     1 request   per ticker scored
#
# So an afternoon of ad-hoc scoring eats the remainder. Scheduling the PERISHABLE
# source last in the queue for a resource that depletes all day is the exact
# inversion of what irreversibility demands: price bars, fundamentals and macro
# are all re-fetchable, while news is gone after NEWS_LOOKBACK_DAYS = 7.
#
# Run this ONCE, from an ordinary (non-admin) PowerShell:
#
#     powershell -ExecutionPolicy Bypass -File scripts\install_daily_task.ps1
#
# Safe to re-run: each task is replaced, never duplicated. Remove them with:
#
#     Unregister-ScheduledTask -TaskName "invest-by-score news collect"
#     Unregister-ScheduledTask -TaskName "invest-by-score daily collect"
#     Unregister-ScheduledTask -TaskName "invest-by-score alerts"

param(
    [string]$NewsTime  = "14:30",
    [string]$DailyTime = "22:00",
    [string]$AlertTime = "22:15"
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

# Wake the machine if asleep, and catch up on a run missed while powered off: a
# laptop shut at the scheduled time is the most likely way for a task to silently
# stop, and StartWhenAvailable turns that into a late run instead of a lost day.
#
# MEASURED 2026-10-06, AND THE COMMENT ABOVE WAS NOT TRUE. The news task missed
# its 14:30 slot (`NumberOfMissedRuns: 1`), was never caught up, and
# `NextRunTime` had already moved to the NEXT weekday -- so Tuesday captured no
# news at all. That night every news alert reported NOT_EVALUATED with the
# honest reason "no news was captured", and the day's perishable articles were
# lost for good: Finnhub serves a rolling window, so a missed day cannot be
# re-fetched later.
#
# TWO DISTINCT GAPS, both fixed here:
#
#   1. -WakeToRun was DOCUMENTED ABOVE BUT NEVER PASSED. The comment claimed
#      the machine would be woken; the settings object had no such flag, so a
#      sleeping machine simply missed the slot. A comment is not a setting.
#
#   2. StartWhenAvailable alone is not enough. Windows only catches up a missed
#      run within a limited window and then gives up until the next scheduled
#      day, which is exactly what happened. A repetition gives the task further
#      chances WITHIN the same day, so a machine powered on at any point before
#      the evening still captures that day's news.
#
# The repetition is harmless when the task already ran: daily_collect is
# idempotent per (ticker, date) -- a second run on the same day re-fetches into
# the same append-only store and the alert path reads by date, not by row count.
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

# -WakeToRun IS NOT SUFFICIENT ON ITS OWN, and that is the second half of the
# same defect. The task flag only ASKS to wake the machine; Windows obeys it
# only where the active power scheme permits wake timers.
#
# MEASURED on this machine (Dell laptop, Balanced scheme) right after setting
# WakeToRun=True on all three tasks:
#
#     RTCWAKE  AC = 0x1 (Enable)     DC = 0x0 (Disable)
#
# So the flag was honoured while plugged in and SILENTLY IGNORED on battery --
# with sleep after 45 minutes on DC, an unplugged laptop would keep missing the
# slot exactly as it did on 2026-10-06, while every task still reported
# WakeToRun=True. Checking the task alone would confirm the wrong thing.
#
# Enabled here for DC too, idempotently, so a reinstall cannot restore the half
# of the fix that lives outside the task definition.
powercfg /setdcvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 1 | Out-Null
powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 1 | Out-Null
powercfg /setactive SCHEME_CURRENT | Out-Null
Write-Host "Wake timers enabled on AC and battery (powercfg RTCWAKE = 1)"
Write-Host ""

# NOTE ON THIS MACHINE'S SLEEP MODEL, since it changes what to expect:
# `powercfg /availablesleepstates` reports S0 Low Power Idle (Modern Standby)
# with S1/S2/S3 unavailable. S0 keeps the network connected and honours wake
# timers, which is BETTER for scheduled work than classic S3. Hibernation would
# block an RTC wake, and HIBERNATEIDLE is 0 on AC and 0x7fffffff on DC -- never,
# in both cases -- so it does not interfere.

function Register-InvestTask {
    param(
        [string]$Name,
        [string]$Script,
        [string]$Arguments,
        [string]$Time,
        [string]$LogName,
        [string]$Description
    )

    $log = Join-Path $repo "data\$LogName"
    $scriptPath = Join-Path $repo $Script

    # cmd /c so the redirection is handled by a shell rather than the scheduler.
    $command = "/c cd /d `"$repo`" && `"$python`" `"$scriptPath`" $Arguments >> `"$log`" 2>&1"

    $action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $command
    $trigger = New-ScheduledTaskTrigger -Weekly `
        -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $Time

    # RETRY WITHIN THE SAME DAY. Without this a single missed slot loses the
    # whole day (MEASURED 2026-10-06, see the settings block above). Repeating
    # every 2 hours for 8 hours means a machine powered on any time between the
    # scheduled hour and 8 hours later still collects that day.
    #
    # Set on the CIM object because New-ScheduledTaskTrigger exposes repetition
    # only for -Once triggers, not for -Weekly.
    $trigger.Repetition = (New-ScheduledTaskTrigger -Once -At $Time `
        -RepetitionInterval (New-TimeSpan -Hours 2) `
        -RepetitionDuration (New-TimeSpan -Hours 8)).Repetition

    # NO -Principal IS PASSED, AND THAT DEFAULT IS LOAD-BEARING.
    #
    # Register-ScheduledTask defaults to LogonType=Interactive ("run only when
    # the user is logged on"), which is what these tasks need: they read the
    # operator's USER-scope environment variables (ALERT_EMAIL_PASSWORD,
    # FINNHUB_API_KEY). A task registered to run "whether the user is logged on
    # or not" runs in a session that does not have them, so email would silently
    # report UNCONFIGURED and news would report no key -- the failure would look
    # like a credential problem rather than a scheduling one.
    #
    # VERIFIED 2026-10-06 that the lock screen does NOT defeat this, which is
    # the obvious worry: the logon session that started 10/05 12:35 was still
    # alive 1d06h later, and the 22:00/22:15 tasks ran at 11:02 PM with
    # result=0 while the machine was unattended. LOCKED, DISPLAY OFF and ASLEEP
    # all keep the session logged on.
    #
    # The cases that DO stop these tasks are different: signing out, switching
    # user, or rebooting and not signing back in. After a restart the operator
    # must sign in once; nothing runs before that.
    Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger `
        -Settings $settings -Description $Description -Force | Out-Null

    Write-Host "Registered '$Name'"
    Write-Host "  runs   : Mon-Fri at $Time"
    Write-Host "  command: $python $scriptPath $Arguments"
    Write-Host "  log    : $log"
    Write-Host ""
}

# 1. NEWS, EARLY. The one irreversible source, ahead of the day's consumption.
Register-InvestTask `
    -Name "invest-by-score news collect" `
    -Script "scripts\daily_collect.py" `
    -Arguments "--sources news,events" `
    -Time $NewsTime `
    -LogName "collect.log" `
    -Description ("Captures PERISHABLE news (gone after 7 days) before the " +
                  "day's NewsAPI quota is spent. MEASURED: a 22:00 slot failed " +
                  "five consecutive nights on quota_exhausted.")

# 2. EVERYTHING RE-FETCHABLE, after the US close.
Register-InvestTask `
    -Name "invest-by-score daily collect" `
    -Script "scripts\daily_collect.py" `
    -Arguments "--sources prices,fundamentals,macro" `
    -Time $DailyTime `
    -LogName "collect.log" `
    -Description ("Captures prices, fundamentals and macro after the US close. " +
                  "All three are re-fetchable, so this slot costs nothing if missed.")

# 3. ALERTS, after collection so the detectors see the day's data.
Register-InvestTask `
    -Name "invest-by-score alerts" `
    -Script "scripts\run_alerts.py" `
    -Arguments "" `
    -Time $AlertTime `
    -LogName "alerts.log" `
    -Description ("Runs every detector, stores findings in the append-only " +
                  "alert ledger for the Monitoring tab, and emails them.")

Write-Host "Verify with:  Get-ScheduledTask -TaskName 'invest-by-score*'"
Write-Host "Run one now:  Start-ScheduledTask -TaskName 'invest-by-score alerts'"
Write-Host ""
Write-Host "NOTE: news collection needs NEWS_PROVIDER_API_KEY in the environment,"
Write-Host "      and it must be set at USER or MACHINE scope -- a variable set"
Write-Host "      only in a terminal session is invisible to the scheduler."
Write-Host ""
Write-Host "      Email needs ALERT_EMAIL_USER and ALERT_EMAIL_PASSWORD (a Gmail"
Write-Host "      App Password, not the account password). Without them every"
Write-Host "      alert is still recorded and visible in the Monitoring tab;"
Write-Host "      only the email channel is disabled."
