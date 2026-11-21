# One-time setup: virtual environment, dependencies, dataset, training.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = $null
foreach ($v in "3.12", "3.11", "3.13", "3.10") {
    try {
        & py "-$v" -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { $py = @("py", "-$v"); break }
    } catch { break }  # Python launcher not installed
}
if (-not $py) { $py = @("python") }
Write-Host "Using $($py -join ' ')"
if (-not (Test-Path ".venv\Scripts\python.exe")) { & $py[0] $py[1..9] -m venv .venv }
$venv = ".\.venv\Scripts\python.exe"
& $venv -m pip install --upgrade pip
& $venv -m pip install -r requirements.txt
& $venv -m cropyield.build_dataset
& $venv -m cropyield.train
Write-Host "`nSetup complete. Start the app with:  .\run.ps1" -ForegroundColor Green
