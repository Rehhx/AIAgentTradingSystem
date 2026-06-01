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

# NO-MARGIN posture (board decision): --max-leverage 1.0 means the vol-target can
# only DE-RISK (scale <=1.0), never borrow -> the book can never be margin-called.
# Trailing stop widened to 20% = catastrophe-only (an 8% stop fought the mean-
# reversion sleeve, which deliberately buys deeper dips). Signal exits do the rest.
# don't trade on US market holidays could be added here; weekends are excluded by the schedule
& $py "runners\daily_rebalance.py" --book portfolio_full --xs-universe sp500 `
      --vol-target 0.17 --max-leverage 1.0 --crypto-sleeve --trail-pct 20 --live *>> $log

"==== exit code $LASTEXITCODE ====" | Out-File -Append -Encoding utf8 $log

# monitoring + track-record: regime-posture check, daily P&L log, drawdown/drift alarms
"==== $(Get-Date -Format 'HH:mm:ss') | monitor / track-record ====" |
    Out-File -Append -Encoding utf8 $log
& $py "runners\monitor.py" *>> $log

# live-vs-backtest tracking dashboard (regenerates TRACKING.md)
"==== $(Get-Date -Format 'HH:mm:ss') | tracking dashboard ====" |
    Out-File -Append -Encoding utf8 $log
& $py "runners\tracking_dashboard.py" *>> $log
