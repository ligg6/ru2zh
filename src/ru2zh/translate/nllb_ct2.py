"""本地 NLLB（ctranslate2）翻译器：俄译中。

重依赖（transformers / ctranslate2）在方法内延迟 import，模块顶层保持轻量。
"""

from __future__ import annotations

from ..textutils import normalize_zh_punct


class NllbTranslator:
    """基于 ctranslate2 + NLLB-200 的俄译中翻译器（惰性加载模型与分词器）。"""

    name = "nllb"

    def __init__(
        self,
        cfg_or_model_dir,
        device: str = "cpu",
        compute_type: str = "int8",
        src_lang: str = "rus_Cyrl",
        tgt_lang: str = "zho_Hans",
        beam_size: int = 4,
    ):
        # 兼容两种构造方式：
        #   1) translate/base.py 的工厂按 NllbTranslator(cfg, device, compute_type) 传入 AppConfig；
        #   2) 直接传入模型目录字符串。
        # 首个参数为 AppConfig 时取其 nllb_model_dir，否则视为模型目录路径。
        from ..config import AppConfig

        if isinstance(cfg_or_model_dir, AppConfig):
            model_dir = cfg_or_model_dir.nllb_model_dir
        else:
            model_dir = cfg_or_model_dir

        self.model_dir = str(model_dir)
        self.device = device
        self.compute_type = compute_type
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.beam_size = beam_size

        self._tok = None  # 惰性加载：分词器
        self._tr = None  # 惰性加载：ctranslate2 翻译器

    def _ensure_loaded(self) -> None:
        """首次调用时加载分词器与模型（延迟 import 重依赖）。"""
        if self._tr is not None:
            return

        from pathlib import Path

        from ..config import anchor_to_root

        # 相对路径在必要时锚定到项目根，保证从任意工作目录运行都能找到模型
        self.model_dir = anchor_to_root(self.model_dir)
        if not Path(self.model_dir).is_dir():
            raise FileNotFoundError(
                f"未找到 NLLB 模型目录：{self.model_dir}。"
                "请先运行 scripts/download_models.py 下载模型。"
            )

        from .. import runtime

        # 加载 ctranslate2 前必须先引导 CUDA DLL（非 Windows 上为空操作）
        runtime.bootstrap_cuda_dlls()

        import os
        import warnings

        # 在首次 import transformers 前压低其日志级别，
        # 避免无 torch 环境下打印 "PyTorch was not found" 的提示干扰用户输出
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

        import transformers
        from transformers import AutoTokenizer

        # transformers 5.13 加载该分词器会打印一条无关的
        # "incorrect regex pattern ... fix_mistral_regex" 警告：对 NLLB 是误报
        # （实测分词正确），这里临时压低日志级别并过滤警告以免干扰用户输出。
        _prev_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._tok = AutoTokenizer.from_pretrained(
                    self.model_dir, src_lang=self.src_lang
                )
        finally:
            transformers.logging.set_verbosity(_prev_verbosity)

        import ctranslate2

        self._tr = ctranslate2.Translator(
            self.model_dir, device=self.device, compute_type=self.compute_type
        )

    def translate_batch(self, texts: list[str]) -> list[str]:
        """批量把俄文翻译为中文，返回与输入等长的译文列表（已规范化中文标点）。"""
        self._ensure_loaded()

        results: list[str] = [""] * len(texts)

        # 空字符串不送模型，直接返回空串
        nonempty_idx: list[int] = []
        batch_tokens: list[list[str]] = []
        for i, t in enumerate(texts):
            if not t or not t.strip():
                continue
            nonempty_idx.append(i)
            batch_tokens.append(self._tok.convert_ids_to_tokens(self._tok.encode(t)))

        if batch_tokens:
            res = self._tr.translate_batch(
                batch_tokens,
                target_prefix=[[self.tgt_lang]] * len(batch_tokens),
                beam_size=self.beam_size,
                max_batch_size=8,
            )
            for j, r in enumerate(res):
                # 去掉输出首个目标语言前缀 token，再解码为文本
                out = self._tok.decode(
                    self._tok.convert_tokens_to_ids(r.hypotheses[0][1:]),
                    skip_special_tokens=True,
                )
                results[nonempty_idx[j]] = normalize_zh_punct(out)

        return results
