@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d %~dp0
setlocal

echo ============================================================
echo   ru2zh 网页界面
echo ============================================================

rem 检查虚拟环境是否存在
if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境 .venv。
    echo   请先双击 install.bat 完成安装，再运行本脚本。
    pause
    exit /b 1
)

set "PYTHONPATH=src"

echo 正在启动网页界面，稍后浏览器会自动打开 http://127.0.0.1:7860
echo 若浏览器未自动弹出，请手动在浏览器地址栏输入上面的网址。
echo 关闭本窗口即可停止服务。
echo.

".venv\Scripts\python.exe" -m ru2zh.webui
if errorlevel 1 (
    echo.
    echo [错误] 网页界面启动失败。请查看上方错误信息，或运行 install.bat 重新安装。
    echo   也可运行 .venv\Scripts\python scripts\check_env.py 进行环境诊断。
    pause
    exit /b 1
)

pause
exit /b 0
