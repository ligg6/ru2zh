#!/usr/bin/env python3
"""下载 ru2zh 所需模型（faster-whisper + NLLB ctranslate2）。

用法：
    python scripts/download_models.py [--dev] [--endpoint URL] [--models-dir DIR]

  --dev          下载小模型（faster-whisper-small + NLLB-600M），用于开发/测试。
  默认（不加 --dev）下载生产模型（faster-whisper-large-v3 + NLLB-1.3B）。
  --endpoint     指定 HuggingFace 端点（如 https://hf-mirror.com）。
  --models-dir   模型保存目录（默认 models）。

本脚本保持独立，不 import ru2zh 包。
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# 开发模型集合
DEV_MODELS = [
    {
        "repo_id": "Systran/faster-whisper-small",
        "revision": None,
        "local_dir": "faster-whisper-small",
        "type": "whisper",
    },
    {
        "repo_id": "entai2965/nllb-200-distilled-600M-ctranslate2",
        "revision": "86876131d0a16b17ced0a5c558fdc3e4613ae545",
        "local_dir": "nllb-200-distilled-600M-ct2",
        "type": "nllb",
    },
]

# 生产模型集合
PROD_MODELS = [
    {
        "repo_id": "Systran/faster-whisper-large-v3",
        "revision": None,
        "local_dir": "faster-whisper-large-v3",
        "type": "whisper",
    },
    {
        "repo_id": "entai2965/nllb-200-distilled-1.3B-ctranslate2",
        "revision": "19b46b1e266e3e385e1286cf67779fc14c541f3d",
        "local_dir": "nllb-200-distilled-1.3B-ct2",
        "type": "nllb",
    },
]

# 各类型模型下载后需要校验的文件
_WHISPER_REQUIRED = ["model.bin", "config.json", "tokenizer.json"]
_WHISPER_VOCAB_ANY = ["vocabulary.txt", "vocabulary.json"]
_NLLB_REQUIRED = ["model.bin", "sentencepiece.bpe.model", "shared_vocabulary.json"]

_MIRROR = "https://hf-mirror.com"


def _probe_huggingface() -> bool:
    """探测能否访问 huggingface.co（HEAD 请求，8 秒超时）。"""
    req = urllib.request.Request("https://huggingface.co", method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=8):
            return True
    except Exception:
        return False


def _select_endpoint(endpoint_arg: str | None) -> None:
    """在 import huggingface_hub 之前决定端点。"""
    if endpoint_arg:
        os.environ["HF_ENDPOINT"] = endpoint_arg
        print(f"[提示] 使用指定端点：{endpoint_arg}")
        return
    if os.environ.get("HF_ENDPOINT"):
        print(f"[提示] 使用环境变量 HF_ENDPOINT：{os.environ['HF_ENDPOINT']}")
        return
    if not _probe_huggingface():
        os.environ["HF_ENDPOINT"] = _MIRROR
        print("[提示] 无法访问 huggingface.co，已切换到国内镜像 hf-mirror.com")


def _dir_size_human(path: Path) -> str:
    """返回目录总大小的人类可读字符串。"""
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    size = float(total)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _verify(spec: dict, local_dir: Path) -> list[str]:
    """校验模型目录必需文件，返回缺失文件名列表。"""
    missing: list[str] = []
    if spec["type"] == "whisper":
        for name in _WHISPER_REQUIRED:
            if not (local_dir / name).is_file():
                missing.append(name)
        if not any((local_dir / n).is_file() for n in _WHISPER_VOCAB_ANY):
            missing.append("vocabulary.txt 或 vocabulary.json")
    else:  # nllb
        for name in _NLLB_REQUIRED:
            if not (local_dir / name).is_file():
                missing.append(name)
    return missing


def _manual_guide(spec: dict, local_dir: Path, error: Exception) -> None:
    """打印手动下载指引。"""
    repo = spec["repo_id"]
    print(f"\n[错误] 下载 {repo} 失败：{error}", file=sys.stderr)
    print("请尝试手动下载：", file=sys.stderr)
    print(f"  官方地址：https://huggingface.co/{repo}", file=sys.stderr)
    print(f"  国内镜像：{_MIRROR}/{repo}", file=sys.stderr)
    print(f"  下载后放入目录：{local_dir}", file=sys.stderr)
    if spec.get("revision"):
        print(f"  注意 revision 需为：{spec['revision']}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="下载 ru2zh 所需模型（faster-whisper + NLLB ctranslate2）。"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="下载开发用小模型（faster-whisper-small + NLLB-600M）。",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="指定 HuggingFace 端点（如 https://hf-mirror.com）。",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="模型保存目录（默认 models）。",
    )
    args = parser.parse_args()

    # 端点选择必须在 import huggingface_hub 之前完成
    _select_endpoint(args.endpoint)

    from huggingface_hub import snapshot_download

    models = DEV_MODELS if args.dev else PROD_MODELS
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n准备下载 {'开发' if args.dev else '生产'}模型，共 {len(models)} 个。")

    completed: list[tuple[str, Path]] = []
    for spec in models:
        local_dir = models_dir / spec["local_dir"]
        print(f"\n=== 正在下载 {spec['repo_id']} → {local_dir} ===")
        try:
            snapshot_download(
                repo_id=spec["repo_id"],
                revision=spec.get("revision"),
                local_dir=str(local_dir),
            )
        except Exception as e:  # noqa: BLE001
            _manual_guide(spec, local_dir, e)
            return 1

        missing = _verify(spec, local_dir)
        if missing:
            print(
                f"[错误] {spec['repo_id']} 下载不完整，缺少文件："
                f"{', '.join(missing)}（目录 {local_dir}）",
                file=sys.stderr,
            )
            return 1

        completed.append((spec["repo_id"], local_dir))

    print("\n========== 下载完成，大小汇总 ==========")
    for repo, local_dir in completed:
        print(f"  {local_dir}  ({_dir_size_human(local_dir)})  <- {repo}")
    print("全部模型下载并校验成功。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
