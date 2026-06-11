# run_web.ps1 — launch the QUANT·DESK backend (serves web/ + runs the REAL pipeline).
# The Control page's one click then runs research -> build -> validate -> deploy
# (DRY-RUN, gated). No live order can be placed from the web UI.
# Opens http://127.0.0.1:8787 automatically. Ctrl+C to stop.
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py (Join-Path $PSScriptRoot "web\server.py")
