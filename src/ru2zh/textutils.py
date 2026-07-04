"""纯文本处理工具（只依赖标准库 + re）。

包含：俄语分句、把 whisper 分段合并为完整句子、中文标点规范化，
以及判断文本是否含西里尔字母 / CJK 字符的小工具。
"""

from __future__ import annotations

import re
import string

from .datatypes import Segment

# ---------------------------------------------------------------------------
# 分句相关
# ---------------------------------------------------------------------------

# 需要保护的俄语常见缩写：其内部/末尾的点号不应触发分句
_ABBREVIATIONS = [
    "и т.д.",
    "и т.п.",
    "т.д.",
    "т.п.",
    "т.е.",
    "т.к.",
    "до н.э.",
    "н.э.",
    "гг.",
    "г.",
    "ул.",
    "др.",
    "пр.",
    "см.",
    "стр.",
    "рис.",
    "им.",
    "проф.",
    "акад.",
    "д.",
    "п.",
]

# 占位符：临时替换被保护的点号，分句后再还原
_DOT_PH = "\x00"

# 缩写匹配正则（要求前面是词边界，避免匹配 "друг." 里的 "г."）
_ABBREV_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")"
)

# 句子边界：句末标点（可重复，如 ?! …）+ 可选闭合引号 + 空白 +（大写字母/«/引号/数字）
_SENT_BOUNDARY_RE = re.compile(
    r'([.!?…]+["»\')]?)\s+(?=[«"“0-9A-ZА-ЯЁ])'
)

# 判断文本是否以句末标点结尾（可后跟闭合引号/空白）
_ENDS_SENTENCE_RE = re.compile(r"[.!?…][»\"'”]*\s*$")

# 多余空白折叠
_WS_RE = re.compile(r"\s+")

_LATIN = set(string.ascii_letters)


def _protect_dots(text: str) -> str:
    """把缩写内、数字小数点中的点号临时替换为占位符。"""

    def _abbr_repl(m: re.Match) -> str:
        return m.group(0).replace(".", _DOT_PH)

    text = _ABBREV_RE.sub(_abbr_repl, text)
    # 数字之间的小数点（如 3.14）
    text = re.sub(r"(?<=\d)\.(?=\d)", _DOT_PH, text)
    return text


def _restore_dots(text: str) -> str:
    return text.replace(_DOT_PH, ".")


def split_sentences_ru(text: str) -> list[str]:
    """按俄语句末标点 .!?… 将文本分成句子列表。

    会保护常见缩写（т.д., г., ул. 等）和数字小数点不被误拆，
    引号内的标点也不会提前断句。断句条件：句末标点后跟空白，
    且下一个非空字符为大写字母/«/引号/数字。
    """
    if not text or not text.strip():
        return []

    protected = _protect_dots(text)

    sentences: list[str] = []
    last = 0
    for m in _SENT_BOUNDARY_RE.finditer(protected):
        chunk = protected[last : m.end(1)]
        sentences.append(chunk)
        last = m.end()  # 跳过分隔空白
    tail = protected[last:]
    if tail:
        sentences.append(tail)

    result = []
    for s in sentences:
        s = _restore_dots(s).strip()
        if s:
            result.append(s)
    return result


def _collapse_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _ends_with_sentence_punct(text: str) -> bool:
    return bool(_ENDS_SENTENCE_RE.search(text))


def merge_segments_to_sentences(segments: list[Segment]) -> list[Segment]:
    """把 whisper 按停顿断开的分段合并为完整句子分段。

    累积文本直到以句末标点结尾；同时设上限防止失控：
    合并段时长 ≤ 30 秒且字符数 ≤ 400，超限即强制切断。
    合并段 start = 首段 start，end = 末段 end，text 用空格连接（折叠多余空白）。
    空文本段跳过。
    """
    result: list[Segment] = []
    buf_texts: list[str] = []
    buf_start: float | None = None
    buf_end: float | None = None

    for seg in segments:
        text = seg.text.strip() if seg.text else ""
        if not text:
            continue
        if buf_start is None:
            buf_start = seg.start
        buf_end = seg.end
        buf_texts.append(text)

        merged_text = _collapse_ws(" ".join(buf_texts))
        duration = buf_end - buf_start
        over_limit = duration > 30 or len(merged_text) > 400

        if _ends_with_sentence_punct(merged_text) or over_limit:
            result.append(Segment(start=buf_start, end=buf_end, text=merged_text))
            buf_texts = []
            buf_start = None
            buf_end = None

    if buf_texts and buf_start is not None and buf_end is not None:
        merged_text = _collapse_ws(" ".join(buf_texts))
        result.append(Segment(start=buf_start, end=buf_end, text=merged_text))

    return result


# ---------------------------------------------------------------------------
# 中文标点规范化
# ---------------------------------------------------------------------------

_PUNCT_MAP = {
    "?": "？",
    "!": "！",
    ";": "；",
}


def _is_latin(ch: str) -> bool:
    return ch in _LATIN


def normalize_zh_punct(text: str) -> str:
    """把中文译文里的英文标点替换为中文标点（保守替换）。

    规则：, → ，   . → 。   ? → ？   ! → ！   : → ：   ; → ；
    成对的英文双引号 "..." → “...”。
    但【数字之间的小数点/逗号不动】（如 3.14、1,000），
    拉丁字母词内的点不动（如 example.com），数字之间的冒号不动（如时间 3:14）。
    """
    if not text:
        return text

    # 先处理成对英文双引号 → 中文引号
    text = re.sub(r'"([^"]*)"', r"“\1”", text)

    out: list[str] = []
    n = len(text)
    for i, ch in enumerate(text):
        prev = text[i - 1] if i > 0 else ""
        nxt = text[i + 1] if i + 1 < n else ""
        if ch == ",":
            if prev.isdigit() and nxt.isdigit():
                out.append(ch)  # 千分位逗号，保留
            else:
                out.append("，")
        elif ch == ".":
            if (prev.isdigit() and nxt.isdigit()) or (
                _is_latin(prev) and _is_latin(nxt)
            ):
                out.append(ch)  # 小数点 / 域名内的点，保留
            else:
                out.append("。")
        elif ch == ":":
            if prev.isdigit() and nxt.isdigit():
                out.append(ch)  # 时间 3:14 等，保留
            else:
                out.append("：")
        elif ch in _PUNCT_MAP:
            out.append(_PUNCT_MAP[ch])
        else:
            out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------------------
# 字符判断小工具
# ---------------------------------------------------------------------------

_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def contains_cyrillic(text: str) -> bool:
    """文本是否含西里尔字母。"""
    return bool(text) and bool(_CYRILLIC_RE.search(text))


def contains_cjk(text: str) -> bool:
    """文本是否含 CJK（中日韩）汉字。"""
    return bool(text) and bool(_CJK_RE.search(text))
