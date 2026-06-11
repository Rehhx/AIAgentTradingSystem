# run_web.ps1 — open the QUANT·DESK web frontend in the default browser.
# Self-contained static app (no server, no build step). Deep-links:
#   web/index.html#dashboard | #agents | #control
Start-Process (Join-Path $PSScriptRoot "web\index.html")
