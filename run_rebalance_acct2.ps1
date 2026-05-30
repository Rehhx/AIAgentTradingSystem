# run_rebalance_acct2.ps1 — ACCOUNT 2: managed-futures (crisis-alpha L/S trend) book.
# The diversified long/SHORT trend program that profits in macro bears (2022 +6.5%
# vs S&P -18%). Runs alongside account 1 (the long-equity growth book).
# REQUIRES: ALPACA_API_KEY_2/SECRET_2 set in .env AND shorting + margin enabled on
# that Alpaca paper account. Uses whole-share orders (no fractional shorts).
$ErrorActionPreference = "Continue"

$proj   = Split-Path -Parent $MyInvocation.MyCommand.Path
$py     = Join-Path $proj ".venv\Scripts\python.exe"
$logdir = Join-Path $proj "logs"
if (-not (Test-Path $logdir)) { New-Item -ItemType Directory -Path $logdir | Out-Null }
$log    = Join-Path $logdir ("rebalance_acct2_" + (Get-Date -Format "yyyy-MM-dd") + ".log")

Set-Location $proj
"==== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | account 2 managed-futures ====" |
    Out-File -Append -Encoding utf8 $log

& $py "runners\daily_rebalance.py" --book managed_futures --account 2 --whole-shares `
      --vol-target 0.12 --max-leverage 1.5 --live *>> $log

"==== exit code $LASTEXITCODE ====" | Out-File -Append -Encoding utf8 $log
