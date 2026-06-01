# register_tasks.ps1 — one-time setup: registers daily rebalance tasks in Windows
# Task Scheduler. Run this ONCE as Administrator. After that the tasks fire
# automatically on weekdays; this machine must be on/awake when they trigger.
#
# Usage (PowerShell as Admin):
#   .\register_tasks.ps1
#
# To verify afterwards:
#   Get-ScheduledTask -TaskName "DailyRebalance_*" | Select TaskName, State

$proj = Split-Path -Parent $MyInvocation.MyCommand.Path
$ps   = "powershell.exe"

$settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 20) `
    -MultipleInstances IgnoreNew

$weekdays = "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"

# --- Account 1: equity growth book (6:15 AM open run) -----------------------
$action_am = New-ScheduledTaskAction -Execute $ps `
    -Argument "-NonInteractive -File `"$proj\run_rebalance.ps1`""
$trigger_am = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At "06:15AM"
Register-ScheduledTask -TaskName "DailyRebalance_AM" `
    -Action $action_am -Trigger $trigger_am -Settings $settings `
    -RunLevel Highest -Force
Write-Host "Registered: DailyRebalance_AM  (6:15 AM weekdays)"

# --- Account 1: equity growth book (3:50 PM close run) ----------------------
$action_pm = New-ScheduledTaskAction -Execute $ps `
    -Argument "-NonInteractive -File `"$proj\run_rebalance_close.ps1`""
$trigger_pm = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At "03:50PM"
Register-ScheduledTask -TaskName "DailyRebalance_Close" `
    -Action $action_pm -Trigger $trigger_pm -Settings $settings `
    -RunLevel Highest -Force
Write-Host "Registered: DailyRebalance_Close  (3:50 PM weekdays)"

# --- Account 2: managed-futures crisis-alpha (6:20 AM open run) -------------
$action_mf = New-ScheduledTaskAction -Execute $ps `
    -Argument "-NonInteractive -File `"$proj\run_rebalance_acct2.ps1`""
$trigger_mf = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At "06:20AM"
Register-ScheduledTask -TaskName "DailyRebalance_Acct2" `
    -Action $action_mf -Trigger $trigger_mf -Settings $settings `
    -RunLevel Highest -Force
Write-Host "Registered: DailyRebalance_Acct2  (6:20 AM weekdays)"

Write-Host ""
Write-Host "Done. Verify with:"
Write-Host "  Get-ScheduledTask -TaskName 'DailyRebalance_*' | Select TaskName, State"
