"""运行时相关：CUDA DLL 引导、设备探测、设备/精度解析、HF 镜像端点设置。

重要：本模块【绝不在 import 时加载 ctranslate2】，所有 import ctranslate2
的地方都必须先调用 bootstrap_cuda_dlls()。
"""

from __future__ import annotations

import os
import sys

from .config import AppConfig

# 幂等标志与已添加的 DLL 目录缓存
_bootstrapped = False
_added_dirs: list[str] = []


def bootstrap_cuda_dlls() -> list[str]:
    """在 Windows 上把 pip 安装的 nvidia cublas/cudnn 的 bin 目录加入 DLL 搜索路径。

    非 Windows 平台直接返回 []。幂等（重复调用不会重复添加）。
    必须在 import ctranslate2 之前调用。返回本次（或首次）添加的目录列表。
    """
    global _bootstrapped, _added_dirs
    if sys.platform != "win32":
        return []
    if _bootstrapped:
        return list(_added_dirs)

    import site
    import sysconfig

    search_roots: list[str] = []
    try:
        search_roots.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            search_roots.append(user_site)
    except Exception:
        pass
    for key in ("purelib", "platlib"):
        p = sysconfig.get_paths().get(key)
        if p:
            search_roots.append(p)

    subdirs = (
        os.path.join("nvidia", "cublas", "bin"),
        os.path.join("nvidia", "cudnn", "bin"),
    )

    added: list[str] = []
    seen: set[str] = set()
    for root in search_roots:
        for sub in subdirs:
            d = os.path.join(root, sub)
            if d in seen:
                continue
            seen.add(d)
            if os.path.isdir(d):
                try:
                    os.add_dll_directory(d)  # 仅 Windows 存在
                except Exception:
                    pass
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                added.append(d)

    _added_dirs = added
    _bootstrapped = True
    return added


def detect_cuda_device_count() -> int:
    """返回可用的 CUDA 设备数量；任何异常都返回 0。"""
    try:
        bootstrap_cuda_dlls()
        import ctranslate2  # 延迟 import

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def resolve_device(cfg: AppConfig) -> tuple[str, str]:
    """根据配置解析最终的 (device, compute_type)。

    device：auto 时自动选 cuda（有 GPU）或 cpu；显式 cuda/cpu 尊重用户。
    require_gpu：None 时按平台推断（Windows→True，其他→False）；生效为 True 且
        最终落到 cpu 且用户未显式指定 cpu → 抛 RuntimeError（含详细中文排查提示）。
    compute_type：auto 时按 device/engine 推断；显式值尊重用户。
    """
    device = cfg.device
    user_explicit_cpu = device == "cpu"

    if device == "auto":
        device = "cuda" if detect_cuda_device_count() > 0 else "cpu"

    require_gpu = cfg.require_gpu
    if require_gpu is None:
        require_gpu = sys.platform == "win32"

    if require_gpu and device == "cpu" and not user_explicit_cpu:
        raise RuntimeError(
            "未检测到可用的 NVIDIA GPU / CUDA。\n"
            "请依次排查：\n"
            "  1. 确认已安装最新的 NVIDIA 显卡驱动；\n"
            "  2. 确认已运行 install.bat 安装 nvidia-cublas-cu12 与 nvidia-cudnn-cu12；\n"
            "  3. 可运行 python scripts/check_env.py 进行诊断；\n"
            "如确实要用 CPU 运行，请加 --cpu 参数，"
            "或在 config.yaml 中设置 require_gpu: false。"
        )

    compute_type = cfg.compute_type
    if compute_type == "auto":
        if device == "cuda":
            # 统一用 int8_float16：与 float16 精度差异可忽略，
            # 且切换翻译引擎时 whisper 无需按不同精度重复加载，
            # 8GB 显存下与 NLLB 共存也安全（约 3GB + 2GB）。
            compute_type = "int8_float16"
        else:
            compute_type = "int8"

    return device, compute_type


def apply_hf_endpoint(cfg: AppConfig) -> None:
    """若配置了 hf_endpoint 且环境未设 HF_ENDPOINT，则设置之。

    必须在任何 huggingface_hub import 之前调用。
    """
    if cfg.hf_endpoint and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = cfg.hf_endpoint
