# run_dashboard.ps1 — launch the interactive control panel in your browser.
$proj = Split-Path -Parent $MyInvocation.MyCommand.Path
$py   = Join-Path $proj ".venv\Scripts\python.exe"
Set-Location $proj
& $py -m streamlit run "app\dashboard.py"
