"""核心数据类型定义（不依赖任何重依赖）。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Segment:
    """一段转写文本（单语，通常来自 ASR）。"""

    start: float  # 起始时间（秒）
    end: float  # 结束时间（秒）
    text: str  # 文本内容


@dataclass
class BilingualSegment:
    """一段双语文本（俄文 + 中文译文）。"""

    start: float  # 起始时间（秒）
    end: float  # 结束时间（秒）
    ru: str  # 俄文原文
    zh: str  # 中文译文


@dataclass
class TranscriptResult:
    """一次完整转写 + 翻译的结果。"""

    audio_path: str  # 源音频文件路径
    segments: list[BilingualSegment]  # 双语分段列表
    # 元信息：duration（时长）、whisper_model（模型）、engine（翻译引擎）、elapsed（耗时）等
    meta: dict = field(default_factory=dict)
