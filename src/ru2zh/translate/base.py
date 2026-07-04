"""翻译器协议与工厂函数。

工厂通过函数内延迟 import 具体实现，因此本模块导入时不依赖任何重依赖，
且允许 nllb_ct2 / llm_api 模块在本阶段尚不存在。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..config import AppConfig


@runtime_checkable
class Translator(Protocol):
    """翻译器统一接口。"""

    name: str

    def translate_batch(self, texts: list[str]) -> list[str]:
        """批量把俄文翻译为中文，返回与输入等长的译文列表。"""
        ...


def get_translator(cfg: "AppConfig", device: str, compute_type: str) -> Translator:
    """根据配置的引擎构造对应的翻译器（延迟 import 具体实现）。

    engine == "nllb" → NllbTranslator；
    engine ∈ {claude, openai, deepseek} → LlmApiTranslator；
    其他 → ValueError。
    """
    engine = cfg.engine
    if engine == "nllb":
        from .nllb_ct2 import NllbTranslator  # 延迟 import（本阶段模块可不存在）

        return NllbTranslator(cfg, device, compute_type)
    if engine in ("claude", "openai", "deepseek"):
        from .llm_api import LlmApiTranslator  # 延迟 import

        return LlmApiTranslator(cfg)
    raise ValueError(
        f"未知的翻译引擎：{engine!r}（可选值：nllb / claude / openai / deepseek）"
    )
