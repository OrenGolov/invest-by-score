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
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

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
