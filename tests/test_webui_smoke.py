"""Web 界面端到端冒烟测试（@pytest.mark.slow）。

用开发小模型（faster-whisper-small + NLLB-600M，CPU + int8）真实启动 Gradio 服务，
再用 gradio_client 调用 /process 接口，验证返回的状态与俄中对照结果。

需要 models/faster-whisper-small、models/nllb-200-distilled-600M-ct2 与
tests/data/ru_short.mp3。默认被 pytest 跳过，用 `-m slow` 显式运行。
首次运行会加载 CPU 模型，较慢属正常；内存有限，请串行运行，勿并行。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ru2zh import textutils
from ru2zh.config import load_config
from ru2zh.webui import build_app

pytestmark = pytest.mark.slow

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _REPO_ROOT / "models"
_NLLB_DIR = _MODELS_DIR / "nllb-200-distilled-600M-ct2"
_WHISPER_DIR = _MODELS_DIR / "faster-whisper-small"
_AUDIO_SHORT = _REPO_ROOT / "tests" / "data" / "ru_short.mp3"

_PORT = 7861


def _iter_strings(obj):
    """递归遍历 client 返回的任意嵌套结构，产出其中所有字符串（用于检查汉字）。"""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)


def test_webui_end_to_end(monkeypatch):
    # 前置资源检查
    assert _AUDIO_SHORT.is_file(), f"缺少测试音频：{_AUDIO_SHORT}"
    assert _NLLB_DIR.is_dir(), f"缺少 NLLB 模型：{_NLLB_DIR}"
    assert _WHISPER_DIR.is_dir(), f"缺少 whisper 模型：{_WHISPER_DIR}"

    # 用开发小模型配置覆盖（绝对路径，避免受运行目录影响）
    monkeypatch.setenv("RU2ZH_WHISPER_MODEL", "small")
    monkeypatch.setenv("RU2ZH_MODELS_DIR", str(_MODELS_DIR))
    monkeypatch.setenv("RU2ZH_NLLB_MODEL_DIR", str(_NLLB_DIR))
    monkeypatch.setenv("RU2ZH_DEVICE", "cpu")
    monkeypatch.setenv("RU2ZH_COMPUTE_TYPE", "int8")
    monkeypatch.setenv("RU2ZH_REQUIRE_GPU", "0")
    monkeypatch.setenv("RU2ZH_PORT", str(_PORT))

    from gradio_client import Client, handle_file

    cfg = load_config()
    app = build_app(cfg)
    # Gradio 6：launch() 不再接受 show_api，故不传（API 文档可见性由事件级 api_visibility 控制）
    app.launch(
        prevent_thread_lock=True,
        server_name="127.0.0.1",
        server_port=_PORT,
        inbrowser=False,
    )
    try:
        client = Client(f"http://127.0.0.1:{_PORT}", verbose=False)
        status, table, files = client.predict(
            handle_file(str(_AUDIO_SHORT)),  # 音频（filepath 输入）
            "nllb",  # 翻译引擎
            "",       # API 密钥（本地引擎留空）
            "small",  # whisper 模型
            "",       # API 模型名
            "",       # 自定义 base_url
            5,        # beam_size
            api_name="/process",
        )
    finally:
        app.close()

    # ---------- 断言状态 ----------
    assert isinstance(status, str)
    assert "❌" not in status, f"状态区出现错误：{status}"
    assert "完成" in status, f"状态区未显示完成：{status}"

    # 段数 > 0
    m = re.search(r"段数：(\d+)", status)
    assert m is not None, f"状态区未包含段数信息：{status}"
    assert int(m.group(1)) > 0, f"段数应大于 0：{status}"

    # ---------- 断言表格含汉字（结构在不同 client 版本下可能是 dict/list，做容错遍历） ----------
    assert table, "结果表不应为空"
    has_cjk = any(textutils.contains_cjk(s) for s in _iter_strings(table))
    has_cyrillic = any(textutils.contains_cyrillic(s) for s in _iter_strings(table))
    assert has_cjk, f"结果表未包含中文翻译：{table!r}"
    assert has_cyrillic, f"结果表未包含俄文原文：{table!r}"
