"""ASR（语音识别）引擎：基于 faster-whisper 的俄语转写。

重依赖（faster_whisper）在方法内延迟 import，模块顶层保持轻量。
构建 WhisperModel 前必须先调 runtime.bootstrap_cuda_dlls()。
"""

from __future__ import annotations

from typing import Callable

from .config import AppConfig
from .datatypes import Segment


class WhisperEngine:
    """封装 faster-whisper 的 WhisperModel，惰性加载、可复用。"""

    def __init__(self, cfg: AppConfig, device: str, compute_type: str):
        # 保存配置与设备信息；模型在首次 transcribe 时才真正构建（惰性加载）
        self.cfg = cfg
        self.device = device
        self.compute_type = compute_type
        self._model = None  # 惰性加载占位

    def _ensure_loaded(self) -> None:
        """首次调用时构建 WhisperModel（延迟 import 重依赖）。"""
        if self._model is not None:
            return

        import os

        from . import config as config_mod
        from . import runtime

        # 构建 WhisperModel 前必须先引导 CUDA DLL（非 Windows 上为空操作）
        runtime.bootstrap_cuda_dlls()

        from faster_whisper import WhisperModel  # 延迟 import，保持模块轻量

        model_path = config_mod.resolve_whisper_model(self.cfg)
        kwargs = {"device": self.device, "compute_type": self.compute_type}
        if self.device == "cpu":
            # CPU 上放开线程数，加快转写
            kwargs["cpu_threads"] = os.cpu_count() or 1
        self._model = WhisperModel(model_path, **kwargs)

    def transcribe(
        self,
        audio_path: str,
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> tuple[list[Segment], dict]:
        """把音频转写为俄语分段列表，返回 (segments, meta)。

        meta 含 duration、language、language_probability、whisper_model。
        progress_cb(比例0~1, 当前段文本) 每识别出一段调用一次。
        """
        from . import config as config_mod

        cfg = self.cfg
        self._ensure_loaded()

        seg_iter, info = self._model.transcribe(
            audio_path,
            language="ru",
            beam_size=cfg.beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=cfg.vad_min_silence_ms),
            condition_on_previous_text=False,
            initial_prompt=cfg.initial_prompt,
        )

        # info.duration 可能为 0，进度计算需防除零
        duration = float(getattr(info, "duration", 0.0) or 0.0)

        segments: list[Segment] = []
        for seg in seg_iter:
            text = (seg.text or "").strip()
            if not text:
                continue  # 跳过空白文本段
            segments.append(
                Segment(start=float(seg.start), end=float(seg.end), text=text)
            )
            if progress_cb is not None:
                frac = min(seg.end / duration, 1.0) if duration else 0.0
                progress_cb(frac, text)

        meta = {
            "duration": duration,
            "language": getattr(info, "language", "ru"),
            "language_probability": getattr(info, "language_probability", None),
            "whisper_model": config_mod.resolve_whisper_model(cfg),
        }
        return segments, meta
