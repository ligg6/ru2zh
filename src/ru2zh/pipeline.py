"""编排层：把音频转写 + 翻译串成一条流水线。

提供 transcribe_and_translate 主入口，并对 WhisperEngine 与翻译器做进程内缓存，
以便 Web UI 等场景连续处理多个文件时不重复加载模型。重依赖均在下层模块内延迟 import。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from .config import AppConfig
from .datatypes import BilingualSegment, TranscriptResult

# ---------------------------------------------------------------------------
# 引擎缓存（键 = 影响模型加载的参数元组）
# ---------------------------------------------------------------------------

_whisper_cache: dict = {}
_translator_cache: dict = {}

# 每批翻译的最大句子数
_TRANSLATE_BATCH_SIZE = 16


def clear_engines() -> None:
    """清空 WhisperEngine 与翻译器缓存（释放已加载的模型引用）。"""
    _whisper_cache.clear()
    _translator_cache.clear()


def _get_whisper_engine(cfg: AppConfig, device: str, compute_type: str):
    """取（或构建并缓存）WhisperEngine。键包含解析后的模型路径与设备/精度。

    同一进程最多只保留一个 WhisperEngine：换模型/设备/精度时先逐出旧实例，
    避免多份 whisper 模型同时驻留显存/内存导致 OOM。
    """
    import gc

    from . import config as config_mod
    from .asr import WhisperEngine

    key = (config_mod.resolve_whisper_model(cfg), device, compute_type)
    engine = _whisper_cache.get(key)
    if engine is None:
        if _whisper_cache:
            _whisper_cache.clear()
            gc.collect()  # 尽快释放被逐出模型占用的显存/内存
        engine = WhisperEngine(cfg, device, compute_type)
        _whisper_cache[key] = engine
    else:
        # 复用已加载模型，但刷新配置以尊重本次的 beam_size / vad / initial_prompt
        engine.cfg = cfg
    return engine


def _translator_key(cfg: AppConfig, device: str, compute_type: str):
    """构造翻译器缓存键（只含影响加载/连接的参数）。"""
    from . import config as config_mod

    if cfg.engine == "nllb":
        return ("nllb", cfg.nllb_model_dir, device, compute_type)
    # LLM API 引擎：与设备无关，按引擎 + 模型 + 端点 + 密钥区分
    # （密钥必须参与键，否则界面上更换密钥后会命中旧客户端不生效）
    return (cfg.engine, cfg.api_model, cfg.api_base_url, config_mod.resolve_api_key(cfg))


def _get_translator(cfg: AppConfig, device: str, compute_type: str):
    """取（或构建并缓存）翻译器。

    本地 NLLB 翻译器最多只保留一个（换目录/设备/精度时逐出旧实例，释放显存）；
    LLM API 翻译器不占显存，可按键并存。
    """
    import gc

    from .translate.base import get_translator

    key = _translator_key(cfg, device, compute_type)
    translator = _translator_cache.get(key)
    if translator is None:
        if key[0] == "nllb":
            stale = [k for k in _translator_cache if k[0] == "nllb"]
            if stale:
                for k in stale:
                    del _translator_cache[k]
                gc.collect()
        translator = get_translator(cfg, device, compute_type)
        _translator_cache[key] = translator
    return translator


def transcribe_and_translate(
    audio_path: str,
    cfg: AppConfig,
    progress_cb: Callable[[float, str], None] | None = None,
) -> TranscriptResult:
    """转写并翻译单个音频，返回 TranscriptResult。

    进度阶段：转写占 0~0.7，翻译占 0.7~1.0。progress_cb(比例, 中文阶段描述)。
    音频不存在 → FileNotFoundError；无任何转写段 → 返回空 segments 的结果
    （meta["note"]="未检测到语音"），不抛异常。
    """
    from . import runtime
    from . import textutils

    if not Path(audio_path).is_file():
        raise FileNotFoundError(f"音频文件不存在：{audio_path}")

    start_time = time.time()
    device, compute_type = runtime.resolve_device(cfg)

    def _report(frac: float, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(max(0.0, min(frac, 1.0)), msg)

    # ---------- 转写：进度 0~0.7 ----------
    engine = _get_whisper_engine(cfg, device, compute_type)

    def _asr_cb(frac: float, text: str) -> None:
        _report(frac * 0.7, f"转写中：{text}")

    segments, asr_meta = engine.transcribe(str(audio_path), _asr_cb)
    duration = float(asr_meta.get("duration", 0.0) or 0.0)

    # ---------- 无语音：返回空结果，不抛异常 ----------
    if not segments:
        elapsed = time.time() - start_time
        _report(1.0, "未检测到语音")
        meta = _build_meta(
            cfg, device, compute_type, duration, elapsed, segment_count=0
        )
        meta["note"] = "未检测到语音"
        return TranscriptResult(audio_path=str(audio_path), segments=[], meta=meta)

    # ---------- 合并为完整句子 ----------
    _report(0.7, "整理句子…")
    sentences = textutils.merge_segments_to_sentences(segments)

    # ---------- 翻译：进度 0.7~1.0 ----------
    translator = _get_translator(cfg, device, compute_type)
    ru_texts = [s.text for s in sentences]
    zh_texts: list[str] = []
    total = len(ru_texts)
    for i in range(0, total, _TRANSLATE_BATCH_SIZE):
        batch = ru_texts[i : i + _TRANSLATE_BATCH_SIZE]
        zh_texts.extend(translator.translate_batch(batch))
        done = min(i + _TRANSLATE_BATCH_SIZE, total)
        frac = 0.7 + 0.3 * (done / total if total else 1.0)
        _report(frac, f"翻译中… {done}/{total}")

    # ---------- 组装双语分段（中文标点再规范化一遍，兼容 LLM 引擎输出）----------
    bilingual: list[BilingualSegment] = []
    for seg, zh in zip(sentences, zh_texts):
        bilingual.append(
            BilingualSegment(
                start=seg.start,
                end=seg.end,
                ru=seg.text,
                zh=textutils.normalize_zh_punct(zh),
            )
        )

    elapsed = time.time() - start_time
    _report(1.0, "完成")
    meta = _build_meta(
        cfg, device, compute_type, duration, elapsed, segment_count=len(bilingual)
    )
    return TranscriptResult(audio_path=str(audio_path), segments=bilingual, meta=meta)


def _build_meta(
    cfg: AppConfig,
    device: str,
    compute_type: str,
    duration: float,
    elapsed: float,
    segment_count: int,
) -> dict:
    """构造结果 meta。同时写入 elapsed 与 elapsed_seconds（前者供 exporters 显示耗时）。"""
    return {
        "whisper_model": cfg.whisper_model,
        "engine": cfg.engine,
        "device": device,
        "compute_type": compute_type,
        "duration": duration,
        "elapsed": elapsed,
        "elapsed_seconds": elapsed,
        "segment_count": segment_count,
    }
