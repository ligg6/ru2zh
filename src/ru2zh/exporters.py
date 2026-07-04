"""导出器：把 TranscriptResult 输出为 txt / srt / json（纯标准库）。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from .datatypes import TranscriptResult


def format_timestamp(seconds: float) -> str:
    """把秒数格式化为 SRT 时间戳 "HH:MM:SS,mmm"（负数按 0 处理）。"""
    if seconds is None or seconds < 0:
        seconds = 0
    total_ms = int(round(seconds * 1000))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, ms = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _hms(seconds: float) -> str:
    """把秒数格式化为 "HH:MM:SS"（不含毫秒，用于 txt 头部与时间轴）。"""
    if seconds is None or seconds < 0:
        seconds = 0
    total = int(round(seconds))
    hours, total = divmod(total, 3600)
    minutes, secs = divmod(total, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def to_txt(r: TranscriptResult) -> str:
    """输出为带头部信息的纯文本。"""
    lines: list[str] = []
    lines.append(f"音频文件：{Path(r.audio_path).name}")

    duration = r.meta.get("duration")
    if duration is None and r.segments:
        duration = r.segments[-1].end
    if duration is not None:
        lines.append(f"时长：{_hms(duration)}")
    if r.meta.get("whisper_model") is not None:
        lines.append(f"Whisper 模型：{r.meta['whisper_model']}")
    if r.meta.get("engine") is not None:
        lines.append(f"翻译引擎：{r.meta['engine']}")
    if r.meta.get("elapsed") is not None:
        try:
            lines.append(f"耗时：{float(r.meta['elapsed']):.1f} 秒")
        except (TypeError, ValueError):
            lines.append(f"耗时：{r.meta['elapsed']}")

    lines.append("=" * 40)
    lines.append("")

    for seg in r.segments:
        lines.append(f"[{_hms(seg.start)} → {_hms(seg.end)}]")
        lines.append(seg.ru)
        lines.append(seg.zh)
        lines.append("")

    return "\n".join(lines)


def to_srt(r: TranscriptResult, lang: Literal["ru", "zh", "both"]) -> str:
    """输出为标准 SRT 字幕。lang="both" 时字幕为两行：俄文 + 中文。"""
    if lang not in ("ru", "zh", "both"):
        raise ValueError(f"未知的字幕语言：{lang}（应为 ru|zh|both）")

    blocks: list[str] = []
    for i, seg in enumerate(r.segments, start=1):
        ts = f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}"
        if lang == "ru":
            body = seg.ru
        elif lang == "zh":
            body = seg.zh
        else:
            body = f"{seg.ru}\n{seg.zh}"
        blocks.append(f"{i}\n{ts}\n{body}")

    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_json(r: TranscriptResult) -> str:
    """输出为 JSON（ensure_ascii=False，缩进 2）。"""
    return json.dumps(asdict(r), ensure_ascii=False, indent=2)


# 格式 → (后缀, 生成函数)
def _srt_ru(r: TranscriptResult) -> str:
    return to_srt(r, "ru")


def _srt_zh(r: TranscriptResult) -> str:
    return to_srt(r, "zh")


def _srt_both(r: TranscriptResult) -> str:
    return to_srt(r, "both")


_FORMAT_TABLE = {
    "txt": (".txt", to_txt),
    "srt_ru": (".ru.srt", _srt_ru),
    "srt_zh": (".zh.srt", _srt_zh),
    "srt_bilingual": (".bi.srt", _srt_both),
    "json": (".json", to_json),
}


def export_all(
    r: TranscriptResult, out_dir: Path, formats: list[str]
) -> list[Path]:
    """按 formats 导出多种格式到 out_dir，返回生成的文件路径列表。

    文件名 = 音频文件名去扩展名 + 对应后缀。未知格式打印中文警告并跳过。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(r.audio_path).stem

    written: list[Path] = []
    for fmt in formats:
        entry = _FORMAT_TABLE.get(fmt)
        if entry is None:
            print(f"[导出警告] 未知的输出格式：{fmt}，已跳过。")
            continue
        suffix, func = entry
        path = out_dir / f"{stem}{suffix}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(func(r))
        written.append(path)
    return written
