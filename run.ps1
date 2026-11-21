# Starts the FastAPI backend + dashboard on http://localhost:8000
Set-Location $PSScriptRoot
$venv = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $venv)) { Write-Host "Run .\setup.ps1 first." -ForegroundColor Yellow; exit 1 }
if (-not (Test-Path "models\model_bundle.joblib")) { & $venv -m cropyield.train }
Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile", "-Command", "Start-Sleep 5; Start-Process http://localhost:8000"
Write-Host "Dashboard: http://localhost:8000    API docs: http://localhost:8000/docs    Stop: Ctrl+C" -ForegroundColor Green
& $venv -m uvicorn api.main:app --host 127.0.0.1 --port 8000
