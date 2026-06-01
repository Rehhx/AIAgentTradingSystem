# run_rebalance_close.ps1 — 3:50 PM close-run: re-checks signals using near-close
# prices so any intraday signal flip (SPY breaks 200d, early-warning triggers, PEAD
# window expires) is acted on TODAY rather than waiting for tomorrow's open.
# Same command as the morning run; yfinance daily bars include today's partial bar
# when the market is open, so SPY / vol checks reflect the actual intraday price.
$ErrorActionPreference = "Continue"

$proj   = Split-Path -Parent $MyInvocation.MyCommand.Path
$py     = Join-Path $proj ".venv\Scripts\python.exe"
$logdir = Join-Path $proj "logs"
if (-not (Test-Path $logdir)) { New-Item -ItemType Directory -Path $logdir | Out-Null }
$log    = Join-Path $logdir ("rebalance_close_" + (Get-Date -Format "yyyy-MM-dd") + ".log")

Set-Location $proj
"==== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | close-run rebalance ====" |
    Out-File -Append -Encoding utf8 $log

& $py "runners\daily_rebalance.py" --book portfolio_full --xs-universe sp500 `
      --vol-target 0.17 --max-leverage 1.8 --crypto-sleeve --trail-pct 8 --live *>> $log

"==== exit code $LASTEXITCODE ====" | Out-File -Append -Encoding utf8 $log

# monitor + tracking after the close run too
"==== $(Get-Date -Format 'HH:mm:ss') | monitor / track-record ====" |
    Out-File -Append -Encoding utf8 $log
& $py "runners\monitor.py" *>> $log

"==== $(Get-Date -Format 'HH:mm:ss') | tracking dashboard ====" |
    Out-File -Append -Encoding utf8 $log
& $py "runners\tracking_dashboard.py" *>> $log
