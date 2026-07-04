#!/usr/bin/env python3
"""ru2zh 一键安装的主逻辑（由 install.bat 调用）。

为什么安装逻辑用 Python 而不写在批处理里：
cmd.exe 解析含中文（多字节 UTF-8）的批处理时，call/goto 的文件偏移可能错位，
把脚本行从中间截断执行（已在 Windows 11 上实际复现，报错形如
"'pip_install' is not recognized ..."）。因此 install.bat 只保留纯 ASCII 的
引导逻辑，所有安装步骤与中文提示都由本脚本完成——Python 在 Windows 控制台
输出中文走 WriteConsoleW，不受代码页影响。

用法：
    python scripts/install_windows.py [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = ROOT / ".venv"
TUNA_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"

_TOTAL_STEPS = 6

# --dry-run 时只打印将要执行的命令，不实际执行（供开发环境自检）
DRY_RUN = False


def _run(cmd: list[str]) -> int:
    """执行命令并返回退出码。"""
    print("  $ " + subprocess.list2cmdline(cmd), flush=True)
    if DRY_RUN:
        return 0
    return subprocess.call(cmd)


def _venv_python() -> Path:
    """返回虚拟环境内 python 解释器的路径（按当前平台）。"""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _step(n: int, msg: str) -> None:
    print(f"\n[{n}/{_TOTAL_STEPS}] {msg}", flush=True)


def _pip_install(vpy: str, req: Path, desc: str) -> bool:
    """用 pip 安装 requirements 文件；官方源失败自动改用清华镜像重试一次。"""
    if _run([vpy, "-m", "pip", "install", "-r", str(req)]) == 0:
        return True
    print(f"  [提示] {desc}从官方源安装失败，改用清华镜像重试：{TUNA_INDEX}")
    return _run([vpy, "-m", "pip", "install", "-r", str(req), "-i", TUNA_INDEX]) == 0


def main(argv: list[str] | None = None) -> int:
    global DRY_RUN
    parser = argparse.ArgumentParser(description="ru2zh 一键安装主逻辑")
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印将要执行的命令，不实际执行。"
    )
    args = parser.parse_args(argv)
    DRY_RUN = args.dry_run

    os.chdir(ROOT)  # download_models.py 等按项目根目录写 models/

    print("=" * 60)
    print("  ru2zh 一键安装（俄语语音转写 + 俄译中）")
    print("=" * 60)
    print("步骤：创建虚拟环境 -> 安装依赖 -> 环境诊断 -> 下载模型（约 8GB）")
    print("安装过程全自动，中途请勿关闭窗口。")

    if sys.version_info < (3, 10):
        print(f"[错误] 需要 Python 3.10 及以上版本，当前为 {sys.version.split()[0]}。")
        return 1

    # ---------- 1：虚拟环境 ----------
    _step(1, "准备虚拟环境 .venv ...")
    vpy = _venv_python()
    if vpy.exists():
        print("  检测到已存在的 .venv，跳过创建。")
    else:
        if _run([sys.executable, "-m", "venv", str(VENV_DIR)]) != 0:
            print("[错误] 创建虚拟环境失败，请确认磁盘空间充足、Python 安装完整。")
            return 1
        print("  虚拟环境创建完成。")
    vpy_s = str(vpy)

    # ---------- 2：升级 pip（失败不致命，旧版 pip 通常也能装）----------
    _step(2, "升级 pip ...")
    if _run([vpy_s, "-m", "pip", "install", "--upgrade", "pip"]) != 0:
        print(f"  [提示] 升级失败，改用清华镜像重试：{TUNA_INDEX}")
        if _run([vpy_s, "-m", "pip", "install", "--upgrade", "pip", "-i", TUNA_INDEX]) != 0:
            print("  [注意] pip 升级失败，继续用现有版本安装（一般不影响）。")

    # ---------- 3：运行依赖 ----------
    _step(3, "安装运行依赖（requirements.txt）...")
    if not _pip_install(vpy_s, ROOT / "requirements.txt", "运行依赖"):
        print("[错误] 运行依赖安装失败（官方源与清华镜像均失败），请检查网络后重跑 install.bat。")
        return 1

    # ---------- 4：Windows CUDA 运行库 ----------
    _step(4, "安装 CUDA 运行库（requirements-cuda-win.txt）...")
    if sys.platform == "win32":
        print("  提示：nvidia-cudnn-cu12 轮子约 700MB，下载慢属正常现象，请耐心等待。")
        if not _pip_install(vpy_s, ROOT / "requirements-cuda-win.txt", "CUDA 运行库"):
            print("[错误] CUDA 运行库安装失败（官方源与清华镜像均失败），请检查网络后重跑 install.bat。")
            return 1
    else:
        print("  非 Windows 系统，跳过（CUDA 轮子仅用于 Windows）。")

    # ---------- 5：环境诊断（失败不退出，仅提醒）----------
    _step(5, "环境诊断（check_env.py）...")
    if _run([vpy_s, str(ROOT / "scripts" / "check_env.py")]) != 0:
        print("  [注意] 环境诊断发现问题，请留意上方输出（模型尚未下载属正常，将在下一步下载）。")

    # ---------- 6：下载模型 ----------
    _step(6, "下载模型（faster-whisper-large-v3 + NLLB-1.3B，约 8GB）...")
    print("  提示：下载时间较长，支持断点续传；无法访问 huggingface 时会自动切换国内镜像 hf-mirror.com。")
    if _run([vpy_s, str(ROOT / "scripts" / "download_models.py")]) != 0:
        print("[错误] 模型下载失败或不完整。可重新运行 install.bat 继续下载（支持断点续传）。")
        return 1

    print()
    print("=" * 60)
    print("  安装完成！")
    print("  现在可以双击 start_webui.bat 启动网页界面，")
    print("  或双击 transcribe.bat 使用命令行批量转写。")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
