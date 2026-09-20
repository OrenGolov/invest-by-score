# Register the daily collection task with Windows Task Scheduler.
#
# The collector only helps if it actually runs. MEASURED before it existed,
# 2-3 of 15 business days were missing from the W6 ledger because collection
# happened only when somebody remembered.
#
# Run this ONCE, from an ordinary (non-admin) PowerShell:
#
#     powershell -ExecutionPolicy Bypass -File scripts\install_daily_task.ps1
#
# It registers a weekday task at 22:00 local time (after the US close), logs
# to data\collect.log, and is safe to re-run: the task is replaced, never
# duplicated. Remove it with:
#
#     Unregister-ScheduledTask -TaskName "invest-by-score daily collect"

param(
    [string]$Time = "22:00",
    [string]$TaskName = "invest-by-score daily collect"
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$log = Join-Path $repo "data\collect.log"
$script = Join-Path $repo "scripts\daily_collect.py"

# cmd /c so the redirection is handled by a shell rather than the scheduler.
$command = "/c cd /d `"$repo`" && `"$python`" `"$script`" >> `"$log`" 2>&1"

$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $command
$trigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $Time

# Wake the machine if asleep, and catch up on a run missed while powered off:
# a laptop shut at 22:00 is the most likely way for this task to silently
# stop, and StartWhenAvailable is what turns that into a late run instead of
# a lost day.
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Description `
    "Captures perishable market data (news is gone after 7 days) into the W6 ledger." `
    -Force | Out-Null

Write-Host "Registered '$TaskName'"
Write-Host "  runs   : Mon-Fri at $Time"
Write-Host "  command: $python $script"
Write-Host "  log    : $log"
Write-Host ""
Write-Host "Verify with:  Get-ScheduledTask -TaskName '$TaskName'"
Write-Host "Run it now :  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
Write-Host "NOTE: news collection needs NEWSAPI_KEY in the environment."
Write-Host "      Without it the task still captures prices, fundamentals and"
Write-Host "      macro, but every day of news is lost permanently and no"
Write-Host "      OBSERVED event memory can ever be recorded."
