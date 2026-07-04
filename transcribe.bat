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

if "%~1"=="" (
    ".venv\Scripts\python.exe" -m ru2zh.cli --help
    pause
    exit /b 0
)

".venv\Scripts\python.exe" -m ru2zh.cli %*
if errorlevel 1 (
    echo.
    echo [ERROR] Transcription failed. See messages above.
    pause
    exit /b 1
)
echo.
pause
exit /b 0
