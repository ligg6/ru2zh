@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d %~dp0
setlocal

echo ============================================================
echo   ru2zh 一键安装（俄语语音转写 + 俄译中）
echo ============================================================
echo 本脚本将：查找 Python -^> 创建虚拟环境 -^> 安装依赖 -^> 下载模型
echo 首次运行需要联网，模型约 8GB，请耐心等待。
echo.

rem ---------------------------------------------------------------
rem 步骤 1：查找 Python（要求版本 ^>= 3.10）
rem ---------------------------------------------------------------
echo [1/7] 正在查找 Python 3.10 及以上版本...
set "PY="
for %%P in ("py -3.12" "py -3.11" "py -3.10" "py -3" "python") do (
    if not defined PY (
        %%~P -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
        if not errorlevel 1 set "PY=%%~P"
    )
)
if not defined PY (
    echo.
    echo [错误] 未找到 Python 3.10 及以上版本！
    echo   请到官网下载安装：https://www.python.org/downloads/windows/
    echo   安装时务必勾选 "Add python.exe to PATH"（把 Python 加入 PATH）。
    echo   装好后重新双击本脚本 install.bat。
    pause
    exit /b 1
)
echo   已找到 Python：%PY%
echo.

rem ---------------------------------------------------------------
rem 步骤 2：创建虚拟环境 .venv（已存在则跳过）
rem ---------------------------------------------------------------
echo [2/7] 准备虚拟环境 .venv ...
if exist ".venv\Scripts\python.exe" (
    echo   检测到已存在的 .venv，跳过创建。
) else (
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败，请确认磁盘空间充足、Python 安装完整。
        pause
        exit /b 1
    )
    echo   虚拟环境创建完成。
)
set "VENV_PY=.venv\Scripts\python.exe"
echo.

rem ---------------------------------------------------------------
rem 步骤 3：升级 pip
rem ---------------------------------------------------------------
echo [3/7] 正在升级 pip ...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [错误] 升级 pip 失败，请检查网络连接后重试。
    pause
    exit /b 1
)
echo.

rem ---------------------------------------------------------------
rem 步骤 4：安装通用运行依赖（失败自动改用清华镜像重试）
rem ---------------------------------------------------------------
echo [4/7] 正在安装运行依赖（requirements.txt）...
call :pip_install requirements.txt "运行依赖"
if errorlevel 1 (
    echo [错误] 运行依赖安装失败（官方源与清华镜像均失败），请检查网络后重跑 install.bat。
    pause
    exit /b 1
)
echo.

rem ---------------------------------------------------------------
rem 步骤 5：安装 Windows CUDA 运行库（cuDNN 轮子约 700MB）
rem ---------------------------------------------------------------
echo [5/7] 正在安装 CUDA 运行库（requirements-cuda-win.txt）...
echo   提示：nvidia-cudnn-cu12 轮子约 700MB，下载慢属正常现象，请耐心等待。
call :pip_install requirements-cuda-win.txt "CUDA 运行库"
if errorlevel 1 (
    echo [错误] CUDA 运行库安装失败（官方源与清华镜像均失败），请检查网络后重跑 install.bat。
    pause
    exit /b 1
)
echo.

rem ---------------------------------------------------------------
rem 步骤 6：环境诊断（失败不退出，仅提醒）
rem ---------------------------------------------------------------
echo [6/7] 正在进行环境诊断（check_env.py）...
"%VENV_PY%" scripts\check_env.py
if errorlevel 1 (
    echo   [注意] 环境诊断发现问题，请留意上方输出（模型尚未下载属正常，将在下一步下载）。
)
echo.

rem ---------------------------------------------------------------
rem 步骤 7：下载模型（生产模型 large-v3 + NLLB-1.3B，约 8GB）
rem ---------------------------------------------------------------
echo [7/7] 正在下载模型（faster-whisper-large-v3 + NLLB-1.3B，约 8GB）...
echo   提示：下载时间较长，支持断点续传；若无法访问 huggingface，脚本会自动切换到国内镜像 hf-mirror.com。
"%VENV_PY%" scripts\download_models.py
if errorlevel 1 (
    echo [错误] 模型下载失败或不完整。可重新运行 install.bat 继续下载（支持断点续传）。
    pause
    exit /b 1
)
echo.

echo ============================================================
echo   安装完成！
echo   现在可以双击 start_webui.bat 启动网页界面。
echo   或双击 transcribe.bat 使用命令行批量转写。
echo ============================================================
pause
exit /b 0

rem ===============================================================
rem 子程序：pip 安装，失败时自动改用清华镜像重试一次
rem   %1 = requirements 文件名   %2 = 中文描述
rem ===============================================================
:pip_install
"%VENV_PY%" -m pip install -r %1
if not errorlevel 1 exit /b 0
echo   [提示] %~2 从官方源安装失败，改用清华镜像重试：
echo          https://pypi.tuna.tsinghua.edu.cn/simple
"%VENV_PY%" -m pip install -r %1 -i https://pypi.tuna.tsinghua.edu.cn/simple
if not errorlevel 1 exit /b 0
exit /b 1
