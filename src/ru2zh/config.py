"""应用配置：AppConfig 数据类与加载/解析辅助函数。

配置来源优先级（从低到高）：
    默认值 < config.yaml < 环境变量（RU2ZH_*）
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class AppConfig:
    """全局应用配置。"""

    # ---------- ASR（语音识别）----------
    whisper_model: str = "large-v3"  # 尺寸别名或本地模型目录路径
    device: str = "auto"  # auto|cuda|cpu
    compute_type: str = "auto"  # auto|float16|int8_float16|int8
    beam_size: int = 5
    vad_min_silence_ms: int = 500
    initial_prompt: str | None = None
    require_gpu: bool | None = None  # None=自动（Windows 上 True，其他平台 False）

    # ---------- 翻译 ----------
    engine: str = "nllb"  # nllb|claude|openai|deepseek
    nllb_model_dir: str = "models/nllb-200-distilled-1.3B-ct2"
    api_model: str | None = None  # None → 该引擎默认模型
    api_base_url: str | None = None  # 自定义 OpenAI 兼容端点
    api_key: str | None = None  # 一般留空，从环境变量取

    # ---------- 通用 ----------
    models_dir: str = "models"
    hf_endpoint: str | None = None
    output_formats: list[str] = field(
        default_factory=lambda: ["txt", "srt_ru", "srt_zh", "srt_bilingual", "json"]
    )


# 项目根目录：config.py 的上上级目录（src/ru2zh/config.py → 项目根）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _project_root() -> Path:
    """返回项目根目录。"""
    return _PROJECT_ROOT


def _coerce_yaml_value(field_name: str, value, field_type) -> object:
    """把 yaml 读入的值按字段类型做必要转换（主要处理 null → None）。"""
    # yaml 的 null 会被解析成 None，直接保留
    return value


def load_config(path: str | Path | None = None) -> AppConfig:
    """加载配置。

    path 为 None 时依次查找 ./config.yaml、项目根/config.yaml；
    都不存在则返回纯默认值。yaml 中的未知键会被忽略并打印中文警告到 stderr。
    yaml 值 null 会保持为 None。最后应用环境变量覆盖（优先级最高）。
    """
    cfg = AppConfig()
    known = {f.name for f in fields(AppConfig)}

    # 1) 定位 yaml 文件
    yaml_path: Path | None = None
    if path is not None:
        candidate = Path(path)
        if candidate.is_file():
            yaml_path = candidate
    else:
        for candidate in (Path.cwd() / "config.yaml", _project_root() / "config.yaml"):
            if candidate.is_file():
                yaml_path = candidate
                break

    # 2) 读取 yaml 并覆盖默认值
    if yaml_path is not None:
        import yaml  # 延迟 import，保持模块轻量

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            print(
                f"[配置警告] 配置文件 {yaml_path} 顶层不是键值映射，已忽略。",
                file=sys.stderr,
            )
            data = {}
        for key, value in data.items():
            if key not in known:
                print(
                    f"[配置警告] 忽略未知配置键：{key}（文件 {yaml_path}）",
                    file=sys.stderr,
                )
                continue
            setattr(cfg, key, value)

    # 3) 环境变量覆盖（RU2ZH_<FIELDNAME 大写>）
    _apply_env_overrides(cfg)
    return cfg


def _apply_env_overrides(cfg: AppConfig) -> None:
    """用 RU2ZH_* 环境变量覆盖配置（原地修改）。"""
    int_fields = {"beam_size", "vad_min_silence_ms"}
    bool_fields = {"require_gpu"}
    for f in fields(AppConfig):
        env_name = "RU2ZH_" + f.name.upper()
        if env_name not in os.environ:
            continue
        raw = os.environ[env_name]
        if f.name in int_fields:
            try:
                setattr(cfg, f.name, int(raw))
            except ValueError:
                print(
                    f"[配置警告] 环境变量 {env_name}={raw!r} 无法转为整数，已忽略。",
                    file=sys.stderr,
                )
        elif f.name in bool_fields:
            setattr(cfg, f.name, raw.strip().lower() in ("1", "true", "yes"))
        elif f.name == "output_formats":
            setattr(cfg, f.name, [s.strip() for s in raw.split(",") if s.strip()])
        else:
            setattr(cfg, f.name, raw)


def anchor_to_root(path_str: str) -> str:
    """把相对路径在必要时锚定到项目根目录。

    绝对路径或相对当前目录已存在的路径原样返回；否则若「项目根/该路径」存在，
    返回锚定后的绝对路径（这样从任意工作目录运行也能找到 models/ 等项目内目录）；
    都不存在时原样返回，交由调用方处理。
    """
    p = Path(path_str)
    if p.is_absolute() or p.exists():
        return path_str
    rooted = _project_root() / p
    if rooted.exists():
        return str(rooted)
    return path_str


def resolve_whisper_model(cfg: AppConfig) -> str:
    """解析 whisper 模型的实际加载路径或别名。

    - 若 cfg.whisper_model 是已存在的目录路径（含锚定到项目根后存在）→ 返回该路径。
    - 若是别名且 <models_dir>/faster-whisper-<别名> 目录存在且含 model.bin → 返回本地路径。
    - 否则返回别名（faster-whisper 会自行下载，受 HF_ENDPOINT 影响）。
    """
    name = cfg.whisper_model
    p = Path(anchor_to_root(name))
    if p.is_dir():
        return str(p)
    local_dir = Path(anchor_to_root(cfg.models_dir)) / f"faster-whisper-{name}"
    if local_dir.is_dir() and (local_dir / "model.bin").is_file():
        return str(local_dir)
    return name


def resolve_api_key(cfg: AppConfig) -> str | None:
    """解析翻译 API 密钥。

    cfg.api_key 非空则直接使用；否则按引擎从环境变量读取：
        claude → ANTHROPIC_API_KEY
        openai → OPENAI_API_KEY
        deepseek → DEEPSEEK_API_KEY
    （openai 兼容端点也认 OPENAI_API_KEY）
    """
    if cfg.api_key:
        return cfg.api_key
    env_map = {
        "claude": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    env_name = env_map.get(cfg.engine)
    if env_name is None:
        return None
    return os.environ.get(env_name)
