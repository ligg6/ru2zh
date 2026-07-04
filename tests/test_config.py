"""config 模块测试。"""

from __future__ import annotations

from ru2zh.config import (
    AppConfig,
    load_config,
    resolve_api_key,
    resolve_whisper_model,
)

# 可能影响默认值测试的环境变量
_ENV_KEYS = [
    "RU2ZH_ENGINE",
    "RU2ZH_DEVICE",
    "RU2ZH_BEAM_SIZE",
    "RU2ZH_OUTPUT_FORMATS",
    "RU2ZH_WHISPER_MODEL",
    "RU2ZH_REQUIRE_GPU",
]


def _clear_env(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_defaults(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    # 指向一个不存在的路径 → 使用纯默认值
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg.whisper_model == "large-v3"
    assert cfg.device == "auto"
    assert cfg.compute_type == "auto"
    assert cfg.beam_size == 5
    assert cfg.engine == "nllb"
    assert cfg.require_gpu is None
    assert cfg.output_formats == ["txt", "srt_ru", "srt_zh", "srt_bilingual", "json"]
    assert cfg.nllb_model_dir == "models/nllb-200-distilled-1.3B-ct2"


def test_yaml_override(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "whisper_model: small\n"
        "engine: claude\n"
        "beam_size: 3\n"
        "initial_prompt: null\n"
        "require_gpu: false\n",
        encoding="utf-8",
    )
    cfg = load_config(yaml_path)
    assert cfg.whisper_model == "small"
    assert cfg.engine == "claude"
    assert cfg.beam_size == 3
    assert cfg.initial_prompt is None  # null → None
    assert cfg.require_gpu is False
    # 未覆盖的键仍为默认
    assert cfg.device == "auto"


def test_unknown_key_warns_but_survives(tmp_path, monkeypatch, capsys):
    _clear_env(monkeypatch)
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "engine: openai\nnot_a_real_key: 123\n", encoding="utf-8"
    )
    cfg = load_config(yaml_path)
    assert cfg.engine == "openai"
    captured = capsys.readouterr()
    assert "not_a_real_key" in captured.err


def test_env_override(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("RU2ZH_ENGINE", "deepseek")
    monkeypatch.setenv("RU2ZH_BEAM_SIZE", "8")
    monkeypatch.setenv("RU2ZH_OUTPUT_FORMATS", "txt, json ,srt_ru")
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg.engine == "deepseek"
    assert cfg.beam_size == 8
    assert cfg.output_formats == ["txt", "json", "srt_ru"]


def test_env_override_beats_yaml(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("engine: claude\n", encoding="utf-8")
    monkeypatch.setenv("RU2ZH_ENGINE", "openai")
    cfg = load_config(yaml_path)
    assert cfg.engine == "openai"


def test_resolve_api_key_from_cfg():
    cfg = AppConfig(engine="claude", api_key="sk-explicit")
    assert resolve_api_key(cfg) == "sk-explicit"


def test_resolve_api_key_engines(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    assert resolve_api_key(AppConfig(engine="claude", api_key=None)) == "anthropic-key"
    assert resolve_api_key(AppConfig(engine="openai", api_key=None)) == "openai-key"
    assert resolve_api_key(AppConfig(engine="deepseek", api_key=None)) == "deepseek-key"
    # nllb 引擎无对应环境变量
    assert resolve_api_key(AppConfig(engine="nllb", api_key=None)) is None


def test_resolve_api_key_missing_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert resolve_api_key(AppConfig(engine="claude", api_key=None)) is None


def test_resolve_whisper_model_alias_no_local(tmp_path):
    # 别名且本地不存在 → 原样返回别名
    cfg = AppConfig(whisper_model="large-v3", models_dir=str(tmp_path))
    assert resolve_whisper_model(cfg) == "large-v3"


def test_resolve_whisper_model_local_dir(tmp_path):
    # 本地 faster-whisper-<别名> 存在且含 model.bin → 返回本地路径
    local = tmp_path / "faster-whisper-small"
    local.mkdir()
    (local / "model.bin").write_text("x", encoding="utf-8")
    cfg = AppConfig(whisper_model="small", models_dir=str(tmp_path))
    assert resolve_whisper_model(cfg) == str(local)


def test_resolve_whisper_model_explicit_dir(tmp_path):
    # whisper_model 直接是已存在目录 → 原样返回
    d = tmp_path / "my-model"
    d.mkdir()
    cfg = AppConfig(whisper_model=str(d), models_dir=str(tmp_path))
    assert resolve_whisper_model(cfg) == str(d)
