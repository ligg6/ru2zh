#!/usr/bin/env python3
"""备用方案：自行把官方 NLLB 模型转换为 ctranslate2 格式。

仅当预转换仓库（entai2965/nllb-200-distilled-*-ctranslate2）无法下载时才需要本脚本。
转换需要临时安装 torch 与 transformers（CPU 版即可），转换完成后可卸载 torch。

用法：
    python scripts/convert_nllb.py [--size 1.3B|600M] [--models-dir models]

步骤：
    1. pip install torch --index-url https://download.pytorch.org/whl/cpu
    2. python scripts/convert_nllb.py
    3.（可选）pip uninstall torch
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_SIZES = {
    "1.3B": ("facebook/nllb-200-distilled-1.3B", "nllb-200-distilled-1.3B-ct2"),
    "600M": ("facebook/nllb-200-distilled-600M", "nllb-200-distilled-600M-ct2"),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把官方 NLLB 模型转换为 ctranslate2 格式（备用方案）。"
    )
    parser.add_argument("--size", choices=list(_SIZES), default="1.3B", help="模型规格（默认 1.3B）。")
    parser.add_argument("--models-dir", default="models", help="模型保存目录（默认 models）。")
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "[错误] 转换需要 torch（CPU 版即可）。请先执行：\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "转换完成后可执行 pip uninstall torch 卸载。",
            file=sys.stderr,
        )
        return 1

    converter = shutil.which("ct2-transformers-converter")
    if converter is None:
        print("[错误] 未找到 ct2-transformers-converter 命令（应随 ctranslate2 一起安装）。", file=sys.stderr)
        return 1

    repo_id, local_name = _SIZES[args.size]
    out_dir = Path(args.models_dir) / local_name
    print(f"开始转换 {repo_id} → {out_dir}（int8 量化，转换约需数 GB 内存与磁盘）...")

    cmd = [
        converter,
        "--model", repo_id,
        "--output_dir", str(out_dir),
        "--quantization", "int8",
        "--copy_files", "tokenizer_config.json", "special_tokens_map.json",
        "sentencepiece.bpe.model", "tokenizer.json",
        "--force",
    ]
    ret = subprocess.call(cmd)
    if ret != 0:
        print("[错误] 转换失败，请检查上方输出（网络、内存、磁盘空间）。", file=sys.stderr)
        return ret

    print(f"转换完成：{out_dir}\n请确认 config.yaml 的 nllb_model_dir 指向该目录。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
