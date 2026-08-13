# Restart the MSD sweep cleanly.
#   - stops any running sweep and its stage children
#   - archives the current scores.csv so a partial run never destroys history
#   - launches exactly one detached run and reports its PID
#
# The venv python.exe is a launcher stub that re-execs the base interpreter, so
# a healthy single run shows TWO processes in a parent/child pair. That is
# normal and must not be mistaken for two concurrent sweeps.
param(
    [string]$Root = "C:\Dev\Plan_2_FEM_2026",
    [string]$Scratch = "C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad",
    [string]$Tag = "run"
)

$ErrorActionPreference = "Stop"
Set-Location $Root

# 1. stop anything already running
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*run_msd_validation*" -or
                   $_.CommandLine -like "*msd_to_mask*" }
foreach ($p in $procs) {
    try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; Write-Output "stopped $($p.ProcessId)" }
    catch { }
}
Start-Sleep -Seconds 3

# 2. archive whatever scores exist
$scores = Join-Path $Root "results\msd_validation\scores.csv"
if (Test-Path $scores) {
    $dest = Join-Path $Scratch ("msd_scores_" + $Tag + ".csv")
    Copy-Item $scores $dest -Force
    Write-Output "archived scores.csv -> $dest ($((Get-Content $scores).Count) rows)"
}

# 3. launch exactly one detached run
$py  = Join-Path $Root "venv\Scripts\python.exe"
$log = Join-Path $Scratch ("msd_" + $Tag + ".log")
$proc = Start-Process -FilePath $py `
    -ArgumentList @("tools/run_msd_validation.py", "--msd-dir",
                    "data/msd/modified-swiss-dwellings-v2/train") `
    -WorkingDirectory $Root -RedirectStandardOutput $log `
    -RedirectStandardError ($log + ".err") -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 8

$live = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*run_msd_validation*" }
Write-Output ("launched PID " + $proc.Id + "; sweep processes now: " + @($live).Count +
              " (a parent/child pair is one sweep)")
Write-Output ("log: " + $log)
