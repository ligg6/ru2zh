@echo off
rem ru2zh one-click installer (bootstrap only).
rem This file is ASCII-only ON PURPOSE: cmd.exe can mis-parse batch files
rem that contain multi-byte (e.g. Chinese) characters, cutting lines in
rem half. All real install logic and Chinese messages live in
rem scripts\install_windows.py.
setlocal
cd /d %~dp0
set PYTHONUTF8=1

echo ============================================================
echo   ru2zh installer
echo ============================================================
echo Looking for Python 3.10+ ...

set "PY="
for %%P in ("py -3.12" "py -3.11" "py -3.10" "py -3" "python") do (
    if not defined PY (
        %%~P -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
        if not errorlevel 1 set "PY=%%~P"
    )
)
if not defined PY (
    echo.
    echo [ERROR] Python 3.10 or newer was not found.
    echo   Download and install it from:
    echo     https://www.python.org/downloads/windows/
    echo   IMPORTANT: check "Add python.exe to PATH" during setup,
    echo   then run install.bat again.
    echo   See README.md for full instructions in Chinese.
    pause
    exit /b 1
)
echo Found Python: %PY%

%PY% scripts\install_windows.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] Install failed. See messages above.
    pause
    exit /b 1
)
pause
exit /b 0
