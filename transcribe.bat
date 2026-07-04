@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d %~dp0
setlocal

rem 检查虚拟环境是否存在
if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境 .venv。
    echo   请先双击 install.bat 完成安装，再运行本脚本。
    pause
    exit /b 1
)

rem 无参数时打印用法示例
if "%~1"=="" (
    echo ============================================================
    echo   ru2zh 命令行批量转写用法
    echo ============================================================
    echo 用法：transcribe.bat ^<音频文件或文件夹^> [参数]
    echo.
    echo 示例：
    echo   transcribe.bat 录音.mp3
    echo   transcribe.bat D:\录音文件夹 --recursive
    echo   transcribe.bat 录音.mp3 --engine claude
    echo   transcribe.bat 录音.mp3 --formats txt,srt_bilingual
    echo   transcribe.bat 录音.mp3 --cpu
    echo.
    echo 常用参数：
    echo   --cpu           强制使用 CPU 运行（很慢，仅在无 GPU 时使用）
    echo   --engine ^<名^>    翻译引擎：nllb / claude / openai / deepseek
    echo   --formats ^<列表^> 输出格式，逗号分隔：txt,srt_ru,srt_zh,srt_bilingual,json
    echo   --recursive     递归处理子文件夹
    echo.
    echo 也可直接把音频文件或文件夹拖到本 .bat 图标上运行。
    echo ============================================================
    pause
    exit /b 0
)

set "PYTHONPATH=src"

".venv\Scripts\python.exe" -m ru2zh.cli %*
if errorlevel 1 (
    echo.
    echo [错误] 转写过程出错。请查看上方错误信息。
    echo   若提示未检测到 GPU，可加 --cpu 参数用 CPU 运行，或参见 README.md 常见问题。
    pause
    exit /b 1
)

echo.
echo 处理完成。输出文件默认保存在 output 目录中。
pause
exit /b 0
