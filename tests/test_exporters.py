"""exporters 模块测试。"""

from __future__ import annotations

import json

from ru2zh.datatypes import BilingualSegment, TranscriptResult
from ru2zh.exporters import (
    export_all,
    format_timestamp,
    to_json,
    to_srt,
    to_txt,
)


def _sample_result() -> TranscriptResult:
    return TranscriptResult(
        audio_path="/tmp/test.mp3",
        segments=[
            BilingualSegment(0.0, 2.5, "Привет мир", "你好世界"),
            BilingualSegment(2.5, 5.0, "Как дела", "你好吗"),
        ],
        meta={
            "duration": 5.0,
            "whisper_model": "large-v3",
            "engine": "nllb",
            "elapsed": 12.34,
        },
    )


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(3661.5) == "01:01:01,500"
    assert format_timestamp(-5) == "00:00:00,000"


def test_to_srt_both_structure():
    srt = to_srt(_sample_result(), "both")
    lines = srt.splitlines()
    # 第一块：序号、时间轴、俄文、中文
    assert lines[0] == "1"
    assert "-->" in lines[1]
    assert lines[2] == "Привет мир"
    assert lines[3] == "你好世界"
    # 存在第二块序号
    assert "2" in lines


def test_to_srt_single_lang():
    srt_ru = to_srt(_sample_result(), "ru")
    assert "Привет мир" in srt_ru
    assert "你好世界" not in srt_ru


def test_to_txt_contains_both_langs():
    txt = to_txt(_sample_result())
    assert "Привет мир" in txt
    assert "你好世界" in txt
    assert "test.mp3" in txt
    assert "Whisper 模型" in txt


def test_to_json_roundtrip():
    s = to_json(_sample_result())
    data = json.loads(s)
    assert data["audio_path"] == "/tmp/test.mp3"
    assert len(data["segments"]) == 2
    assert data["segments"][0]["ru"] == "Привет мир"
    # ensure_ascii=False → 中文字面出现，而非 \uXXXX 转义
    assert "你好世界" in s


def test_export_all_filenames(tmp_path):
    formats = [
        "txt",
        "srt_ru",
        "srt_zh",
        "srt_bilingual",
        "json",
        "bogus_format",
    ]
    paths = export_all(_sample_result(), tmp_path, formats)
    names = {p.name for p in paths}
    assert names == {
        "test.txt",
        "test.ru.srt",
        "test.zh.srt",
        "test.bi.srt",
        "test.json",
    }
    # 未知格式被跳过
    assert not (tmp_path / "test.bogus_format").exists()
    # 文件确实写入
    for p in paths:
        assert p.is_file()


def test_export_all_creates_dir(tmp_path):
    out = tmp_path / "nested" / "out"
    paths = export_all(_sample_result(), out, ["json"])
    assert len(paths) == 1
    assert paths[0].is_file()
