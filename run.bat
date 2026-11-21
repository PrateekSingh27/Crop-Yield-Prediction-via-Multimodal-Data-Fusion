@echo off
REM Starts the FastAPI backend + dashboard on http://localhost:8000
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" ( echo Run setup.bat first. & exit /b 1 )
if not exist "models\model_bundle.joblib" ".venv\Scripts\python.exe" -m cropyield.train
start "" cmd /c "timeout /t 5 >nul & start http://localhost:8000"
echo Dashboard: http://localhost:8000    API docs: http://localhost:8000/docs    Stop: Ctrl+C
".venv\Scripts\python.exe" -m uvicorn api.main:app --host 127.0.0.1 --port 8000
