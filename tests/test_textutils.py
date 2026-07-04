"""textutils 模块测试。"""

from __future__ import annotations

from ru2zh.datatypes import Segment
from ru2zh.textutils import (
    contains_cjk,
    contains_cyrillic,
    merge_segments_to_sentences,
    normalize_zh_punct,
    split_sentences_ru,
)

# ---------------------------------------------------------------------------
# split_sentences_ru
# ---------------------------------------------------------------------------


def test_split_two_sentences():
    result = split_sentences_ru("Привет. Как дела?")
    assert result == ["Привет.", "Как дела?"]


def test_split_abbreviation_not_split():
    # т.д. 内部的点不应触发分句（否则会被拆成两句）
    result = split_sentences_ru("Купи всё: хлеб, т.д. Потом позвони мне.")
    assert len(result) == 1
    assert "т.д." in result[0]


def test_split_decimal_not_split():
    result = split_sentences_ru("Цена 3.14 доллара. Это дорого.")
    assert len(result) == 2
    assert "3.14" in result[0]


def test_split_ellipsis_and_combo():
    # ?! 组合触发分句，… 结尾不再拆
    result = split_sentences_ru("Правда?! Конечно…")
    assert result == ["Правда?!", "Конечно…"]


def test_split_ellipsis_separator():
    result = split_sentences_ru("Он думал… Потом решил.")
    assert len(result) == 2


def test_split_quotes():
    # «Привет!» — сказал он. 应为一句（引号内 ! 不提前断句）
    result = split_sentences_ru("«Привет!» — сказал он.")
    assert result == ["«Привет!» — сказал он."]


# ---------------------------------------------------------------------------
# merge_segments_to_sentences
# ---------------------------------------------------------------------------


def test_merge_multiple_into_one():
    segs = [
        Segment(0.0, 1.0, "Привет"),
        Segment(1.0, 2.0, "как дела"),
        Segment(2.0, 3.0, "сегодня."),
    ]
    result = merge_segments_to_sentences(segs)
    assert len(result) == 1
    assert result[0].start == 0.0
    assert result[0].end == 3.0
    assert result[0].text == "Привет как дела сегодня."


def test_merge_force_cut_by_duration():
    # 每段 20 秒、无句末标点 → 超 30 秒强制切断
    segs = [
        Segment(0.0, 20.0, "aaa"),
        Segment(20.0, 40.0, "bbb"),
        Segment(40.0, 60.0, "ccc"),
        Segment(60.0, 80.0, "ddd"),
    ]
    result = merge_segments_to_sentences(segs)
    assert len(result) == 2
    assert result[0].start == 0.0
    assert result[0].end == 40.0


def test_merge_force_cut_by_chars():
    # 每段 200 字、无句末标点 → 超 400 字强制切断
    segs = [
        Segment(0.0, 1.0, "a" * 200),
        Segment(1.0, 2.0, "b" * 200),
        Segment(2.0, 3.0, "c" * 200),
    ]
    result = merge_segments_to_sentences(segs)
    assert len(result) == 2


def test_merge_complete_sentences_not_merged():
    segs = [
        Segment(0.0, 1.0, "Привет."),
        Segment(1.0, 2.0, "Как дела?"),
    ]
    result = merge_segments_to_sentences(segs)
    assert len(result) == 2
    assert result[0].text == "Привет."
    assert result[1].text == "Как дела?"


def test_merge_skips_empty():
    segs = [
        Segment(0.0, 1.0, "Привет."),
        Segment(1.0, 2.0, "   "),
        Segment(2.0, 3.0, "Пока."),
    ]
    result = merge_segments_to_sentences(segs)
    assert len(result) == 2
    assert result[0].text == "Привет."
    assert result[1].text == "Пока."


# ---------------------------------------------------------------------------
# normalize_zh_punct
# ---------------------------------------------------------------------------


def test_normalize_basic():
    assert normalize_zh_punct("你好,世界.") == "你好，世界。"


def test_normalize_question_exclaim():
    assert normalize_zh_punct("真的?太好了!") == "真的？太好了！"


def test_normalize_decimal_preserved():
    assert normalize_zh_punct("价格是3.14元") == "价格是3.14元"


def test_normalize_thousands_preserved():
    assert normalize_zh_punct("共1,000人") == "共1,000人"


def test_normalize_domain_preserved():
    assert normalize_zh_punct("访问 example.com 查看") == "访问 example.com 查看"


def test_normalize_paired_quotes():
    assert normalize_zh_punct('他说"你好"') == "他说“你好”"


# ---------------------------------------------------------------------------
# contains_cyrillic / contains_cjk
# ---------------------------------------------------------------------------


def test_contains_cyrillic():
    assert contains_cyrillic("Привет") is True
    assert contains_cyrillic("hello") is False
    assert contains_cyrillic("你好") is False


def test_contains_cjk():
    assert contains_cjk("你好") is True
    assert contains_cjk("Привет") is False
    assert contains_cjk("hello") is False
