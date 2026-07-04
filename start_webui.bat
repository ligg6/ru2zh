@echo off
rem ASCII-only ON PURPOSE (see note in install.bat).
rem Chinese messages are printed by the Python program itself.
setlocal
cd /d %~dp0
set PYTHONUTF8=1

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run install.bat first.
    pause
    exit /b 1
)

set "PYTHONPATH=src"

".venv\Scripts\python.exe" -m ru2zh.webui
if errorlevel 1 (
    echo.
    echo [ERROR] Web UI exited with an error. See messages above.
    echo   Diagnose: .venv\Scripts\python.exe scripts\check_env.py
    pause
    exit /b 1
)
pause
exit /b 0
