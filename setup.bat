@echo off
REM One-time setup: virtual environment, dependencies, dataset, training.
cd /d "%~dp0"
set PY=
for %%V in (3.12 3.11 3.13 3.10) do (
  if not defined PY ( py -%%V -c "import sys" >nul 2>&1 && set PY=py -%%V )
)
if not defined PY set PY=python
echo Using %PY%
if not exist ".venv\Scripts\python.exe" ( %PY% -m venv .venv || goto :fail )
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
".venv\Scripts\python.exe" -m cropyield.build_dataset || goto :fail
".venv\Scripts\python.exe" -m cropyield.train || goto :fail
echo.
echo Setup complete. Start the app with:  run.bat
exit /b 0
:fail
echo.
echo Setup failed - see the error above.
exit /b 1
