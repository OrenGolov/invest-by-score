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

# WHICH INTERPRETER. This used to resolve ".venv\Scripts\python.exe" and fall
# back to bare "python" when absent. MEASURED 2026-10-04, that fallback ALWAYS
# won, because the venv in this repo is "venv" with no leading dot:
#
#     .venv          does not exist
#     venv           exists
#     bare python    pandas 3.0.5, numpy 2.5.2, sklearn MISSING
#     venv python    pandas 3.0.5, numpy 2.5.2, sklearn 1.9.1
#
# The dangerous part is that the fallback LOOKS fine: bare python has pandas,
# so collection succeeds, while anything on the model path dies on a missing
# sklearn - at 22:00, unattended, into a log nobody reads. A scheduled task
# pointing at the wrong interpreter is worse than one that refuses to install.
#
# So both spellings are tried, and the interpreter must PROVE it can import
# what the collector needs before the task is registered at all.
$python = $null
foreach ($candidate in @(
    (Join-Path $repo "venv\Scripts\python.exe"),
    (Join-Path $repo ".venv\Scripts\python.exe")
)) {
    if (Test-Path $candidate) { $python = $candidate; break }
}

if (-not $python) {
    Write-Host "No virtualenv found. Looked for:" -ForegroundColor Red
    Write-Host "  $repo\venv\Scripts\python.exe"
    Write-Host "  $repo\.venv\Scripts\python.exe"
    Write-Host ""
    Write-Host "Create one and install the requirements:"
    Write-Host "  python -m venv venv"
    Write-Host "  venv\Scripts\python.exe -m pip install -r requirements.txt"
    Write-Host ""
    Write-Host "Refusing to register a task against the bare interpreter:"
    Write-Host "it has pandas but not sklearn, so collection would appear to"
    Write-Host "work while the model path failed silently every night."
    exit 1
}

# PROVE IT RUNS THE WORK, rather than trusting the path. The import list is
# what daily_collect.py reaches for, directly or through core/.
Write-Host "Checking $python ..."
$probe = @'
import importlib.util, sys
missing = [m for m in ("pandas", "numpy", "sklearn", "pyarrow")
           if importlib.util.find_spec(m) is None]
if missing:
    print("MISSING:" + ",".join(missing))
    sys.exit(1)
print("OK")
'@
$probeResult = $probe | & $python - 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "That interpreter cannot run the collector: $probeResult" -ForegroundColor Red
    Write-Host ""
    Write-Host "Install the requirements into it:"
    Write-Host "  & `"$python`" -m pip install -r requirements.txt"
    Write-Host ""
    Write-Host "Refusing to register a task that would fail every night."
    exit 1
}
Write-Host "  imports OK (pandas, numpy, sklearn, pyarrow)"

$log = Join-Path $repo "data\collect.log"
$script = Join-Path $repo "scripts\daily_collect.py"
$monitor = Join-Path $repo "scripts\monitor_collection.py"

# cmd /c so the redirection is handled by a shell rather than the scheduler.
#
# THE MONITOR RUNS WITH `&` RATHER THAN `&&`, deliberately. The collector
# exits 1 whenever anything PERISHABLE was lost, which is exactly the case
# the monitor exists to report — chaining with && would skip the alert in
# the only situation that needs one. `&` runs it regardless of exit code.
#
# The monitor's own exit code is what the scheduler records, and it reflects
# COLLECTION health rather than delivery success, so a blocked notification
# channel cannot make a healthy collection look broken.
$command = "/c cd /d `"$repo`" && `"$python`" `"$script`" >> `"$log`" 2>&1 & `"$python`" `"$monitor`" >> `"$log`" 2>&1"

$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $command
$trigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $Time

# Wake the machine if asleep, and catch up on a run missed while powered off:
# a laptop shut at 22:00 is the most likely way for this task to silently
# stop, and StartWhenAvailable is what turns that into a late run instead of
# a lost day.
#
# -WakeToRun WAS MISSING, and the comment above has claimed "wake the machine
# if asleep" since this script was written. MEASURED 2026-10-06 on the
# registered task: WakeToRun=False. A sleeping laptop at 22:00 therefore did
# nothing at all, and StartWhenAvailable only catches the run up at the next
# boot -- by which time the 7-day news window has already eaten the oldest day.
#
# StartWhenAvailable is still the right partner setting, because WakeToRun
# cannot wake a machine that is powered off or hibernated; the two cover
# different failure modes and neither replaces the other.
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
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
Write-Host "NOTE: news collection needs NEWS_PROVIDER_API_KEY in the environment."
Write-Host "      Without it the task still captures prices, fundamentals and"
Write-Host "      macro, but every day of news is lost permanently and no"
Write-Host "      OBSERVED event memory can ever be recorded."
