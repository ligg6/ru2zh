"""在线大模型 API 翻译引擎（俄译中）。

支持三家 provider：claude（Anthropic）、openai、deepseek（OpenAI 兼容端点）。

设计要点：
- anthropic / openai SDK 一律在函数内延迟 import，模块顶层保持轻量；
- 采用「编号行协议」批量翻译：输入编号 1..n 一行一句，要求输出严格同编号同行数，
  便于把译文按行映射回原始位置；
- 构造函数不做任何网络调用，客户端对象首次调用时才构建并缓存，方便测试与界面即时切换；
- 与工厂 get_translator 对接：工厂以 `LlmApiTranslator(cfg)` 方式（传入 AppConfig）
  构造本类，因此 __init__ 第一个参数既接受 provider 字符串，也接受 AppConfig 对象。
"""

from __future__ import annotations

import re
import sys

# ---------------------------------------------------------------------------
# 系统提示词（中文）
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "你是专业的俄译中翻译。输入是俄语口语转写文本，可能含语气词、口头禅、不完整的句子。"
    "要求译文自然流畅、贴合口语语气，符合中文表达习惯。"
    "输入是按行编号的文本（形如「1. ……」「2. ……」）；"
    "输出必须严格保持与输入相同的编号和行数，一行对应一行，编号一一对应。"
    "不要合并或拆分行，不要添加任何解释、注释或额外内容，只输出编号加译文。"
    "专有名词（人名、地名、机构名等）按通用译法翻译。"
)

# 每家 provider 的默认模型
_DEFAULT_MODELS = {
    "claude": "claude-opus-4-8",
    "openai": "gpt-4o",
    "deepseek": "deepseek-chat",
}

# api_key 缺失时提示用户设置的环境变量名
_ENV_VAR_NAMES = {
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

# deepseek 的默认 OpenAI 兼容端点
_DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"


# ---------------------------------------------------------------------------
# 编号行协议：纯函数（便于单元测试）
# ---------------------------------------------------------------------------

# 匹配「编号 + 分隔符 + 文本」的行：允许前导空白，分隔符可为 . ) 、 : ：（含全角）
_NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\s*[.．)）、:：]\s*(.*)$")

# 文本内换行（含 \r）替换为空格用
_NEWLINE_RE = re.compile(r"[\r\n]+")


def build_numbered_block(texts: list[str]) -> str:
    """把文本列表构造成编号行块：``1. <文本>\\n2. <文本>...``。

    文本内部的换行会被替换为空格，保证每条文本占且仅占一行。
    """
    lines: list[str] = []
    for i, text in enumerate(texts, start=1):
        flat = _NEWLINE_RE.sub(" ", text) if text else ""
        lines.append(f"{i}. {flat}")
    return "\n".join(lines)


def parse_numbered_response(response: str, expected: int) -> dict[int, str]:
    """宽容地解析编号行响应，返回 ``{编号: 文本}``。

    解析规则：
    - 接受 ``1.`` / ``1)`` / ``1、`` / ``1:`` 等分隔符，允许前导空白；
    - 忽略 markdown 代码围栏行（以 ``` 开头）和空行；
    - 无编号的行若上一行有有效编号，则视为其续行并拼接到该行；
    - 编号超出 ``[1, expected]`` 范围的整行忽略（也不作为续行归属对象）。
    """
    result: dict[int, str] = {}
    if not response:
        return result

    last_num: int | None = None
    for raw_line in response.splitlines():
        stripped = raw_line.strip()
        # 忽略 markdown 代码围栏与空行
        if not stripped or stripped.startswith("```"):
            continue

        m = _NUMBERED_LINE_RE.match(raw_line)
        if m:
            num = int(m.group(1))
            text = m.group(2).strip()
            if 1 <= num <= expected:
                result[num] = text
                last_num = num
            else:
                # 超范围编号忽略，且清空续行归属，避免误拼接
                last_num = None
            continue

        # 无编号行：若上一行有有效编号则拼接为续行
        if last_num is not None and last_num in result:
            prev = result[last_num]
            result[last_num] = (prev + " " + stripped) if prev else stripped

    return result


def chunk_texts(
    texts: list[str], max_chars: int
) -> list[list[tuple[int, str]]]:
    """按累计字符数把文本分块，每项保留其在原列表中的全局索引。

    - 每块累计字符数尽量不超过 ``max_chars``；
    - 单条文本本身超长时，单独成块（不会与其它条目合并）。

    返回：块列表，每块是 ``[(全局索引, 文本), ...]``。
    """
    chunks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_chars = 0

    for i, text in enumerate(texts):
        length = len(text)
        # 当前块非空且加入本条会超限 → 先收束当前块
        if current and current_chars + length > max_chars:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append((i, text))
        current_chars += length

    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# 翻译器
# ---------------------------------------------------------------------------


class LlmApiTranslator:
    """基于在线大模型 API 的俄译中翻译器。

    构造方式两种（二选一）：
    - 直接指定：``LlmApiTranslator("deepseek", api_key=..., model=...)``；
    - 工厂方式：``LlmApiTranslator(cfg)``，其中 cfg 为 AppConfig（get_translator 使用）。
    """

    def __init__(
        self,
        provider,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        chunk_chars: int = 4000,
        timeout: float = 120.0,
    ):
        # 兼容工厂调用：第一个参数是 AppConfig 时，从中提取各字段
        if not isinstance(provider, str):
            cfg = provider
            from ..config import resolve_api_key  # 延迟 import，避免循环依赖

            provider = cfg.engine
            if api_key is None:
                api_key = resolve_api_key(cfg)
            if model is None:
                model = cfg.api_model
            if base_url is None:
                base_url = cfg.api_base_url

        provider = provider.lower() if provider else provider
        if provider not in _DEFAULT_MODELS:
            raise ValueError(
                f"未知的翻译 provider：{provider!r}（可选值：claude / openai / deepseek）"
            )

        self.provider: str = provider
        self.model: str = model or _DEFAULT_MODELS[provider]

        # deepseek 未显式指定 base_url 时使用默认端点；用户传了则尊重
        if base_url is None and provider == "deepseek":
            base_url = _DEEPSEEK_DEFAULT_BASE_URL
        self.base_url: str | None = base_url

        if not api_key:
            env_var = _ENV_VAR_NAMES[provider]
            raise ValueError(
                f"缺少 {provider} 引擎的 API 密钥。"
                f"请设置环境变量 {env_var}，或在 config.yaml 的 api_key 字段 / 界面中填写。"
            )
        self.api_key: str = api_key

        self.chunk_chars = chunk_chars
        self.timeout = timeout
        self.name = f"{provider}:{self.model}"

        # 客户端首次调用时构建并缓存（构造函数不做任何网络调用）
        self._client = None

    # ------------------------------------------------------------------
    # 批量翻译
    # ------------------------------------------------------------------

    def translate_batch(self, texts: list[str]) -> list[str]:
        """批量把俄文翻译成中文，返回与输入等长、逐条对应的译文列表。

        - 空列表返回 ``[]``；
        - 空字符串（或纯空白）条目不送 API，直接保留原值；
        - 其余按累计字符数分块，逐块调用 _translate_chunk。
        """
        if not texts:
            return []

        results: list[str] = list(texts)  # 默认保留原值（含空字符串条目）

        for chunk in chunk_texts(texts, self.chunk_chars):
            # 过滤掉空字符串 / 纯空白条目，保留全局索引
            sub = [(gi, t) for gi, t in chunk if t and t.strip()]
            if not sub:
                continue
            translated = self._translate_chunk(sub)
            for gi, zh in translated.items():
                results[gi] = zh

        return results

    def _translate_chunk(
        self, chunk: list[tuple[int, str]]
    ) -> dict[int, str]:
        """翻译一个块（块内文本均非空），返回 ``{全局索引: 译文}``。"""
        n = len(chunk)
        # 块内重新编号 1..n，并建立局部编号 → 全局索引的映射
        local_to_global = {i + 1: gi for i, (gi, _) in enumerate(chunk)}
        chunk_only = [t for _, t in chunk]

        user_content = build_numbered_block(chunk_only)

        raw = self._call_api(user_content)
        parsed = parse_numbered_response(raw, n)

        # 行数不符 → 用更严厉的追加指令重试一次
        if len(parsed) != n:
            extra = (
                f"你上次只输出了 {len(parsed)} 行，这是错误的。"
                f"输入共有 {n} 行，你必须恰好输出 {n} 行，"
                f"编号从 1 到 {n}，逐行对应输入，不要多也不要少。"
            )
            raw = self._call_api(user_content, extra_instruction=extra)
            parsed = parse_numbered_response(raw, n)

        result: dict[int, str] = {}
        for local_num in range(1, n + 1):
            gi = local_to_global[local_num]
            if local_num in parsed:
                result[gi] = parsed[local_num]
            else:
                # 重试后仍缺行：回填原俄文并打印中文警告，保证输出长度不变
                original = chunk_only[local_num - 1]
                print(
                    f"[翻译警告] {self.name}：第 {local_num} 行译文缺失，已回填原俄文。",
                    file=sys.stderr,
                )
                result[gi] = original

        return result

    # ------------------------------------------------------------------
    # 底层 API 调用
    # ------------------------------------------------------------------

    def _call_api(self, user_content: str, extra_instruction: str = "") -> str:
        """调用具体 provider 的接口，返回原始文本响应。

        捕获所有异常并转换为 RuntimeError（含 provider / model 与错误摘要），
        便于上层辨别认证失败、限流、网络错误等。
        """
        content = user_content
        if extra_instruction:
            content = f"{user_content}\n\n{extra_instruction}"

        try:
            if self.provider == "claude":
                client = self._get_claude_client()
                resp = client.messages.create(
                    model=self.model,
                    max_tokens=8000,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": content}],
                )
                text = "".join(
                    b.text for b in resp.content if b.type == "text"
                )
            else:
                client = self._get_openai_client()
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                )
                text = resp.choices[0].message.content
        except Exception as e:  # noqa: BLE001 — 统一转为中文 RuntimeError
            raise RuntimeError(
                f"调用 {self.provider} 模型「{self.model}」失败："
                f"{type(e).__name__}: {e}\n"
                "请检查：网络连接是否正常、API 密钥是否有效（认证失败）、"
                "是否触发限流或额度不足（如 429）、base_url 是否正确。"
            ) from e

        return text or ""

    def _get_claude_client(self):
        """构建并缓存 Anthropic 客户端（延迟 import）。"""
        if self._client is None:
            import anthropic  # 延迟 import，保持模块顶层轻量

            kwargs = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def _get_openai_client(self):
        """构建并缓存 OpenAI（兼容）客户端（延迟 import）。"""
        if self._client is None:
            import openai  # 延迟 import

            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client
