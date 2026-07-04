"""llm_api 模块测试（全部离线，不调用真实 API）。"""

from __future__ import annotations

import pytest

from ru2zh.translate.llm_api import (
    LlmApiTranslator,
    build_numbered_block,
    chunk_texts,
    parse_numbered_response,
)


# ---------------------------------------------------------------------------
# 协议纯函数：build_numbered_block
# ---------------------------------------------------------------------------


def test_build_numbered_block_basic():
    assert build_numbered_block(["привет", "как дела"]) == "1. привет\n2. как дела"


def test_build_numbered_block_single():
    assert build_numbered_block(["одна строка"]) == "1. одна строка"


def test_build_numbered_block_newline_flattened():
    # 文本内换行替换为空格
    assert build_numbered_block(["строка1\nстрока2"]) == "1. строка1 строка2"
    assert build_numbered_block(["a\r\nb\nc"]) == "1. a b c"


def test_build_numbered_block_empty_list():
    assert build_numbered_block([]) == ""


# ---------------------------------------------------------------------------
# 协议纯函数：parse_numbered_response
# ---------------------------------------------------------------------------


def test_parse_standard():
    res = parse_numbered_response("1. 你好\n2. 世界", 2)
    assert res == {1: "你好", 2: "世界"}


def test_parse_separator_variants():
    # 变体分隔符 1) / 1、 / 1: / 全角冒号
    res = parse_numbered_response("1) 甲\n2、乙\n3: 丙\n4：丁", 4)
    assert res == {1: "甲", 2: "乙", 3: "丙", 4: "丁"}


def test_parse_leading_whitespace():
    res = parse_numbered_response("   1.   带前导空白\n\t2. 第二行", 2)
    assert res == {1: "带前导空白", 2: "第二行"}


def test_parse_markdown_fence_wrapped():
    text = "```\n1. 一\n2. 二\n```"
    assert parse_numbered_response(text, 2) == {1: "一", 2: "二"}
    # 带语言标注的围栏也应忽略
    text2 = "```text\n1. 一\n2. 二\n```"
    assert parse_numbered_response(text2, 2) == {1: "一", 2: "二"}


def test_parse_out_of_order():
    res = parse_numbered_response("2. 后\n1. 先", 2)
    assert res == {1: "先", 2: "后"}


def test_parse_continuation_join():
    # 无编号行拼接到上一有编号行
    text = "1. 第一句\n续写内容\n2. 第二句"
    assert parse_numbered_response(text, 2) == {1: "第一句 续写内容", 2: "第二句"}


def test_parse_leading_unnumbered_dropped():
    # 前面没有有效编号的无编号行应被丢弃
    text = "开场白\n1. 正文"
    assert parse_numbered_response(text, 1) == {1: "正文"}


def test_parse_out_of_range_ignored():
    res = parse_numbered_response("1. a\n2. b\n3. c", 2)
    assert res == {1: "a", 2: "b"}
    # 超范围行也不应作为续行归属
    res2 = parse_numbered_response("1. a\n5. x\n续行", 2)
    assert res2 == {1: "a"}


def test_parse_empty_response():
    assert parse_numbered_response("", 3) == {}
    assert parse_numbered_response("```\n```", 3) == {}
    assert parse_numbered_response("   \n\n", 3) == {}


# ---------------------------------------------------------------------------
# 协议纯函数：chunk_texts
# ---------------------------------------------------------------------------


def test_chunk_empty_list():
    assert chunk_texts([], 10) == []


def test_chunk_single_over_long():
    long = "a" * 100
    assert chunk_texts([long], 10) == [[(0, long)]]


def test_chunk_over_long_stands_alone():
    a = "a" * 5
    b = "b" * 100
    c = "c" * 5
    chunks = chunk_texts([a, b, c], 10)
    assert chunks == [[(0, a)], [(1, b)], [(2, c)]]


def test_chunk_exactly_full_block():
    # 每条 2 字符，max=4：前两条恰好装满一块，第三条另起一块
    chunks = chunk_texts(["ab", "cd", "ef"], 4)
    assert chunks == [[(0, "ab"), (1, "cd")], [(2, "ef")]]


def test_chunk_preserves_global_indices():
    chunks = chunk_texts(["x", "y", "z"], 100)
    # 全部装进一块，索引保留
    assert chunks == [[(0, "x"), (1, "y"), (2, "z")]]


# ---------------------------------------------------------------------------
# 翻译器：translate_batch（monkeypatch _call_api）
# ---------------------------------------------------------------------------


def _echo_translate(self, user_content, extra_instruction=""):
    """假的 _call_api：把每行俄文加「译:」前缀，编号原样返回。"""
    parsed = parse_numbered_response(user_content, 9999)
    return "\n".join(f"{k}. 译:{parsed[k]}" for k in sorted(parsed))


def test_translate_batch_empty_list():
    tr = LlmApiTranslator("deepseek", api_key="k")
    assert tr.translate_batch([]) == []


def test_translate_batch_corresponds(monkeypatch):
    monkeypatch.setattr(LlmApiTranslator, "_call_api", _echo_translate)
    tr = LlmApiTranslator("deepseek", api_key="k")
    out = tr.translate_batch(["привет", "мир"])
    assert out == ["译:привет", "译:мир"]
    assert len(out) == 2


def test_translate_batch_retry_fills_missing(monkeypatch):
    """第一次响应缺行，重试后补全；_call_api 被调用两次，结果正确。"""
    calls: list[str] = []

    def fake(self, user_content, extra_instruction=""):
        calls.append(extra_instruction)
        parsed = parse_numbered_response(user_content, 9999)
        keys = sorted(parsed)
        if len(calls) == 1:
            keys = keys[:-1]  # 第一次故意少一行
        return "\n".join(f"{k}. 译{k}" for k in keys)

    monkeypatch.setattr(LlmApiTranslator, "_call_api", fake)
    tr = LlmApiTranslator("openai", api_key="k")
    out = tr.translate_batch(["a", "b"])
    assert out == ["译1", "译2"]
    assert len(calls) == 2  # 首次 + 重试


def test_translate_batch_retry_still_missing_backfills(monkeypatch, capsys):
    """重试后仍缺行 → 缺行回填原俄文，并打印中文警告到 stderr。"""

    def fake(self, user_content, extra_instruction=""):
        parsed = parse_numbered_response(user_content, 9999)
        keys = sorted(parsed)[:-1]  # 永远少最后一行
        return "\n".join(f"{k}. 译{k}" for k in keys)

    monkeypatch.setattr(LlmApiTranslator, "_call_api", fake)
    tr = LlmApiTranslator("deepseek", api_key="k")
    out = tr.translate_batch(["первый", "второй"])
    assert len(out) == 2
    assert out[0] == "译1"
    assert out[1] == "второй"  # 回填原俄文
    err = capsys.readouterr().err
    assert "第 2 行" in err
    assert "回填" in err


def test_translate_batch_skips_empty_entries(monkeypatch):
    """空字符串条目不经 _call_api，且在结果中保留为空串。"""
    seen: list[str] = []

    def fake(self, user_content, extra_instruction=""):
        seen.append(user_content)
        parsed = parse_numbered_response(user_content, 9999)
        return "\n".join(f"{k}. 译{k}" for k in sorted(parsed))

    monkeypatch.setattr(LlmApiTranslator, "_call_api", fake)
    tr = LlmApiTranslator("deepseek", api_key="k")
    out = tr.translate_batch(["a", "", "b"])
    assert len(out) == 3
    assert out[1] == ""  # 空条目保留
    assert out[0] == "译1"
    assert out[2] == "译2"
    # 发送给 API 的编号块只含两条（不含空条目）
    assert len(seen) == 1
    assert "a" in seen[0] and "b" in seen[0]
    assert parse_numbered_response(seen[0], 9999) == {1: "a", 2: "b"}


def test_translate_batch_all_empty_no_api(monkeypatch):
    def fake(self, user_content, extra_instruction=""):
        raise AssertionError("空条目不应调用 _call_api")

    monkeypatch.setattr(LlmApiTranslator, "_call_api", fake)
    tr = LlmApiTranslator("openai", api_key="k")
    assert tr.translate_batch(["", "   "]) == ["", "   "]


# ---------------------------------------------------------------------------
# provider 逻辑
# ---------------------------------------------------------------------------


def test_default_models():
    assert LlmApiTranslator("claude", api_key="k").model == "claude-opus-4-8"
    assert LlmApiTranslator("openai", api_key="k").model == "gpt-4o"
    assert LlmApiTranslator("deepseek", api_key="k").model == "deepseek-chat"


def test_explicit_model_and_name():
    tr = LlmApiTranslator("claude", api_key="k", model="claude-custom")
    assert tr.model == "claude-custom"
    assert tr.name == "claude:claude-custom"


def test_deepseek_default_base_url():
    tr = LlmApiTranslator("deepseek", api_key="k")
    assert tr.base_url == "https://api.deepseek.com"


def test_deepseek_respects_user_base_url():
    tr = LlmApiTranslator("deepseek", api_key="k", base_url="http://localhost:11434/v1")
    assert tr.base_url == "http://localhost:11434/v1"


def test_openai_base_url_defaults_none():
    assert LlmApiTranslator("openai", api_key="k").base_url is None


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        LlmApiTranslator("gemini", api_key="k")


def test_missing_api_key_raises_with_env_var_name():
    with pytest.raises(ValueError) as e1:
        LlmApiTranslator("claude", api_key=None)
    assert "ANTHROPIC_API_KEY" in str(e1.value)

    with pytest.raises(ValueError) as e2:
        LlmApiTranslator("openai", api_key=None)
    assert "OPENAI_API_KEY" in str(e2.value)

    with pytest.raises(ValueError) as e3:
        LlmApiTranslator("deepseek", api_key=None)
    assert "DEEPSEEK_API_KEY" in str(e3.value)


def test_constructor_no_network_and_no_client():
    # 构造函数不应构建客户端（无网络调用）
    tr = LlmApiTranslator("claude", api_key="k")
    assert tr._client is None


# ---------------------------------------------------------------------------
# 与工厂 get_translator 集成
# ---------------------------------------------------------------------------


def test_factory_returns_llm_translator(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    from ru2zh.config import AppConfig
    from ru2zh.translate.base import get_translator

    cfg = AppConfig(engine="deepseek")
    tr = get_translator(cfg, "cpu", "int8")
    assert isinstance(tr, LlmApiTranslator)
    assert tr.provider == "deepseek"
    assert tr.model == "deepseek-chat"
    assert tr.base_url == "https://api.deepseek.com"
    assert tr.name == "deepseek:deepseek-chat"
    assert tr.api_key == "deepseek-key"


def test_factory_respects_cfg_model_and_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from ru2zh.config import AppConfig
    from ru2zh.translate.base import get_translator

    cfg = AppConfig(
        engine="openai",
        api_model="gpt-4o-mini",
        api_key="explicit-key",
        api_base_url="https://proxy.example.com/v1",
    )
    tr = get_translator(cfg, "cpu", "int8")
    assert isinstance(tr, LlmApiTranslator)
    assert tr.model == "gpt-4o-mini"
    assert tr.api_key == "explicit-key"
    assert tr.base_url == "https://proxy.example.com/v1"
