@echo off
setlocal

cd /d "%~dp0"

REM --- Choose python command ---
set PY=python
%PY% --version >nul 2>&1
if errorlevel 1 (
  set PY=py
)

REM --- Create venv if not exists ---
if not exist "venv\" (
  echo [1/3] Creating venv...
  %PY% -m venv venv
)

REM --- Activate venv ---
echo [2/3] Activating venv...
call "venv\Scripts\activate.bat"

REM --- Install dependencies ---
echo [3/3] Installing dependencies...
if exist "requirements.txt" (
  pip install -r requirements.txt
) else (
  pip install -r requirements
)

REM --- Run server ---
echo Starting server on http://127.0.0.1:8000
python -m uvicorn main:app --reload

endlocal
pause
