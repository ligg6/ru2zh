#!/usr/bin/env python3
"""ru2zh 环境诊断脚本（中文输出，CPU 环境也可完整运行）。

按序检查：Python/平台、venv、关键包版本、（Windows）CUDA DLL 引导、
CUDA 设备、模型目录、磁盘空间，最后给出总结与退出码。
致命问题（Windows 无 CUDA / 模型缺失）退出码为 1，否则为 0。
"""

import platform
import shutil
import sys
from pathlib import Path

# 注入 src 以便 import ru2zh.runtime
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def _pkg_version(module_name: str) -> str:
    """尝试 import 模块并返回版本字符串，失败返回 "未安装"。"""
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "(已安装，版本未知)")
    except Exception:
        return "未安装"


def _dir_size_human(path: Path) -> str:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    size = float(total)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def main() -> int:
    problems: list[str] = []
    is_windows = sys.platform == "win32"

    print("=" * 56)
    print("ru2zh 环境诊断")
    print("=" * 56)

    # 1) Python 与平台
    print(f"Python 版本：{platform.python_version()}")
    print(f"平台：{sys.platform}  ({platform.platform()})")
    print(f"解释器：{sys.executable}")

    # 2) venv
    expected_venv = (_PROJECT_ROOT / ".venv").resolve()
    try:
        active = Path(sys.prefix).resolve() == expected_venv
    except Exception:
        active = False
    in_any_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if active:
        print("项目 venv：✓ 已激活（.venv）")
    elif in_any_venv:
        print(f"项目 venv：⚠ 处于某个 venv，但不是项目的 .venv（{sys.prefix}）")
    else:
        print("项目 venv：⚠ 未在 venv 中运行")

    # 3) 关键包版本
    print("\n--- 关键包 ---")
    for label, mod in (
        ("faster_whisper", "faster_whisper"),
        ("ctranslate2", "ctranslate2"),
        ("gradio", "gradio"),
        ("transformers", "transformers"),
    ):
        print(f"  {label}: {_pkg_version(mod)}")

    # 4) Windows CUDA DLL 引导
    if is_windows:
        print("\n--- CUDA DLL 引导（Windows）---")
        try:
            from ru2zh import runtime

            added = runtime.bootstrap_cuda_dlls()
            if added:
                for d in added:
                    print(f"  已加入 DLL 目录：{d}")
            else:
                print("  未找到 nvidia cublas/cudnn 的 bin 目录（可能尚未安装）。")
        except Exception as e:  # noqa: BLE001
            print(f"  引导失败：{e}")

    # 5) CUDA 设备
    print("\n--- CUDA 设备 ---")
    try:
        from ru2zh import runtime

        count = runtime.detect_cuda_device_count()
    except Exception as e:  # noqa: BLE001
        count = 0
        print(f"  探测出错（按 0 处理）：{e}")
    if count >= 1:
        print(f"  ✓ 检测到 {count} 个 CUDA 设备，将使用 GPU")
    else:
        if is_windows:
            print("  ✗✗✗ 警告：未检测到 CUDA 设备！✗✗✗")
            print("      排查步骤：")
            print("      1. 安装最新 NVIDIA 显卡驱动；")
            print("      2. 运行 install.bat 安装 nvidia-cublas-cu12 与 nvidia-cudnn-cu12；")
            print("      3. 重新运行本脚本诊断。")
            problems.append("Windows 上未检测到 CUDA 设备")
        else:
            print("  未检测到 CUDA 设备，将使用 CPU（Linux/其他平台属正常情况）。")

    # 6) 模型目录
    print("\n--- 模型目录 ---")
    models_dir = _PROJECT_ROOT / "models"
    found_model = False
    if models_dir.is_dir():
        subdirs = sorted(p for p in models_dir.iterdir() if p.is_dir())
        if subdirs:
            for d in subdirs:
                has_bin = (d / "model.bin").is_file()
                mark = "✓" if has_bin else "⚠(缺 model.bin)"
                print(f"  {mark} {d.name}  ({_dir_size_human(d)})")
                if has_bin:
                    found_model = True
        else:
            print("  （models/ 为空）")
    else:
        print("  （未找到 models/ 目录）")
    if not found_model:
        print("  未检测到已下载的模型，请运行：python scripts/download_models.py")
        problems.append("未检测到已下载的模型")

    # 7) 磁盘空间
    print("\n--- 磁盘空间 ---")
    try:
        usage = shutil.disk_usage(str(_PROJECT_ROOT))
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        print(f"  剩余 {free_gb:.1f} GB / 共 {total_gb:.1f} GB")
        if free_gb < 5:
            print("  ⚠ 磁盘剩余空间较少（<5GB），模型下载可能失败。")
    except Exception as e:  # noqa: BLE001
        print(f"  无法获取磁盘信息：{e}")

    # 8) 总结
    print("\n" + "=" * 56)
    if problems:
        print("发现以下问题：")
        for p in problems:
            print(f"  - {p}")
        print("=" * 56)
        return 1
    print("环境就绪 ✓")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    sys.exit(main())
