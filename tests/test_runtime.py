"""runtime 模块测试。"""

from __future__ import annotations

import sys

import pytest

from ru2zh import runtime
from ru2zh.config import AppConfig


def _patch_cuda(monkeypatch, count: int):
    monkeypatch.setattr(runtime, "detect_cuda_device_count", lambda: count)


# ---------------------------------------------------------------------------
# resolve_device：device 选择
# ---------------------------------------------------------------------------


def test_auto_with_gpu_nllb(monkeypatch):
    _patch_cuda(monkeypatch, 1)
    cfg = AppConfig(device="auto", compute_type="auto", engine="nllb")
    device, compute = runtime.resolve_device(cfg)
    assert device == "cuda"
    assert compute == "int8_float16"


def test_auto_with_gpu_claude(monkeypatch):
    # auto 精度统一为 int8_float16（与引擎无关），避免切换引擎时重复加载 whisper
    _patch_cuda(monkeypatch, 1)
    cfg = AppConfig(device="auto", compute_type="auto", engine="claude")
    device, compute = runtime.resolve_device(cfg)
    assert device == "cuda"
    assert compute == "int8_float16"


def test_auto_no_gpu_cpu(monkeypatch):
    _patch_cuda(monkeypatch, 0)
    # require_gpu=False 避免抛错
    cfg = AppConfig(device="auto", compute_type="auto", engine="nllb", require_gpu=False)
    device, compute = runtime.resolve_device(cfg)
    assert device == "cpu"
    assert compute == "int8"


def test_explicit_cuda_respected(monkeypatch):
    _patch_cuda(monkeypatch, 0)  # 即使检测不到也尊重用户
    cfg = AppConfig(device="cuda", compute_type="auto", engine="nllb")
    device, compute = runtime.resolve_device(cfg)
    assert device == "cuda"
    assert compute == "int8_float16"


def test_explicit_cpu_no_raise(monkeypatch):
    _patch_cuda(monkeypatch, 0)
    monkeypatch.setattr(sys, "platform", "win32")  # 即使 win 且 require 默认 True
    cfg = AppConfig(device="cpu", compute_type="auto", engine="nllb")
    device, compute = runtime.resolve_device(cfg)
    assert device == "cpu"
    assert compute == "int8"


# ---------------------------------------------------------------------------
# resolve_device：require_gpu 逻辑
# ---------------------------------------------------------------------------


def test_require_gpu_true_no_gpu_win32_raises(monkeypatch):
    _patch_cuda(monkeypatch, 0)
    monkeypatch.setattr(sys, "platform", "win32")
    cfg = AppConfig(device="auto", compute_type="auto", engine="nllb", require_gpu=None)
    with pytest.raises(RuntimeError):
        runtime.resolve_device(cfg)


def test_require_gpu_explicit_true_no_gpu_raises(monkeypatch):
    _patch_cuda(monkeypatch, 0)
    monkeypatch.setattr(sys, "platform", "linux")
    cfg = AppConfig(device="auto", compute_type="auto", engine="nllb", require_gpu=True)
    with pytest.raises(RuntimeError):
        runtime.resolve_device(cfg)


def test_require_gpu_none_linux_no_raise(monkeypatch):
    _patch_cuda(monkeypatch, 0)
    monkeypatch.setattr(sys, "platform", "linux")
    cfg = AppConfig(device="auto", compute_type="auto", engine="nllb", require_gpu=None)
    device, compute = runtime.resolve_device(cfg)
    assert device == "cpu"
    assert compute == "int8"


def test_explicit_compute_type_respected(monkeypatch):
    _patch_cuda(monkeypatch, 1)
    cfg = AppConfig(device="auto", compute_type="float16", engine="nllb")
    device, compute = runtime.resolve_device(cfg)
    assert device == "cuda"
    assert compute == "float16"


# ---------------------------------------------------------------------------
# bootstrap_cuda_dlls
# ---------------------------------------------------------------------------


def test_bootstrap_returns_empty_on_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert runtime.bootstrap_cuda_dlls() == []
