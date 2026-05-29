# run_rebalance.ps1 — daily morning rebalance of the `portfolio` book on Alpaca paper.
# Invoked by Windows Task Scheduler (weekdays, before the 6:30am PST open).
# Runs on THIS machine only — the PC must be on/awake when it fires.
$ErrorActionPreference = "Continue"

# project dir = folder this script lives in (portable, no hardcoded path)
$proj   = Split-Path -Parent $MyInvocation.MyCommand.Path
$py     = Join-Path $proj ".venv\Scripts\python.exe"
$logdir = Join-Path $proj "logs"
if (-not (Test-Path $logdir)) { New-Item -ItemType Directory -Path $logdir | Out-Null }
$log    = Join-Path $logdir ("rebalance_" + (Get-Date -Format "yyyy-MM-dd") + ".log")

Set-Location $proj
"==== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | daily rebalance (portfolio) ====" |
    Out-File -Append -Encoding utf8 $log

# don't trade on US market holidays could be added here; weekends are excluded by the schedule
& $py "runners\daily_rebalance.py" --book portfolio_div --xs-universe sp500 `
      --vol-target 0.16 --max-leverage 1.6 --live *>> $log

"==== exit code $LASTEXITCODE ====" | Out-File -Append -Encoding utf8 $log
