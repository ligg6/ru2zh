"""端到端冒烟测试（@pytest.mark.slow）：用开发小模型跑真实转写 + 翻译。

需要 models/faster-whisper-small 与 models/nllb-200-distilled-600M-ct2，
以及 tests/data/ru_short.mp3。默认被 pytest 跳过，用 `-m slow` 显式运行。
内存有限，串行运行，勿并行。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ru2zh import textutils
from ru2zh.config import AppConfig
from ru2zh.exporters import export_all
from ru2zh.pipeline import transcribe_and_translate

pytestmark = pytest.mark.slow

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
_MODELS_DIR = _REPO_ROOT / "models"
_NLLB_DIR = _MODELS_DIR / "nllb-200-distilled-600M-ct2"
_AUDIO_SHORT = _REPO_ROOT / "tests" / "data" / "ru_short.mp3"


def _dev_cfg() -> AppConfig:
    """开发小模型配置（CPU + int8）。用绝对路径以免受运行目录影响。"""
    return AppConfig(
        whisper_model="small",
        models_dir=str(_MODELS_DIR),
        nllb_model_dir=str(_NLLB_DIR),
        engine="nllb",
        device="cpu",
        compute_type="int8",
        require_gpu=False,
    )


@pytest.fixture(scope="module")
def result():
    """整个模块共享一次真实流水线结果（引擎缓存后各测试复用已加载模型）。"""
    assert _AUDIO_SHORT.is_file(), f"缺少测试音频：{_AUDIO_SHORT}"
    assert _NLLB_DIR.is_dir(), f"缺少 NLLB 模型：{_NLLB_DIR}"
    return transcribe_and_translate(str(_AUDIO_SHORT), _dev_cfg())


def test_end_to_end_content(result):
    # 有转写内容
    assert result.segments, "转写结果不应为空"
    # 至少一段俄文含西里尔字母，且对应中文含汉字
    assert any(textutils.contains_cyrillic(seg.ru) for seg in result.segments)
    assert any(textutils.contains_cjk(seg.zh) for seg in result.segments)
    # 每段有俄文就应有对应译文
    for seg in result.segments:
        if textutils.contains_cyrillic(seg.ru):
            assert textutils.contains_cjk(seg.zh), f"未翻译：{seg.ru!r} -> {seg.zh!r}"
    # 时长为正
    assert result.meta["duration"] > 0


def test_export_all_formats(result, tmp_path):
    formats = ["txt", "srt_ru", "srt_zh", "srt_bilingual", "json"]
    paths = export_all(result, tmp_path, formats)
    assert len(paths) == 5
    for p in paths:
        assert p.is_file()
        assert p.stat().st_size > 0, f"输出文件为空：{p}"

    # json 可被解析
    json_path = next(p for p in paths if p.suffix == ".json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["segments"]

    # srt 文件首行是 "1"
    for p in paths:
        if p.name.endswith(".srt"):
            with open(p, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            assert first_line == "1", f"{p} 首行应为 '1'，实际为 {first_line!r}"


def test_cli_smoke(tmp_path):
    """通过子进程跑 CLI，验证退出码与输出文件。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SRC_DIR)
    # config.yaml 默认指向生产 NLLB（未下载），用环境变量覆盖到开发小模型
    env["RU2ZH_NLLB_MODEL_DIR"] = str(_NLLB_DIR)
    env["RU2ZH_MODELS_DIR"] = str(_MODELS_DIR)

    out_dir = tmp_path / "cli_out"
    cmd = [
        sys.executable,
        "-m",
        "ru2zh.cli",
        str(_AUDIO_SHORT),
        "-o",
        str(out_dir),
        "--model",
        "small",
        "--cpu",
        "--formats",
        "txt,json",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"CLI 退出码非 0：\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    stem = _AUDIO_SHORT.stem
    assert (out_dir / f"{stem}.txt").is_file()
    assert (out_dir / f"{stem}.json").is_file()
    assert (out_dir / f"{stem}.txt").stat().st_size > 0
