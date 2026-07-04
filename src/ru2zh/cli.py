"""命令行入口：批量把俄语音频转写并翻译成中文，导出 txt/srt/json。

用法：
    python -m ru2zh.cli <文件或目录>... [-o OUT_DIR] [--engine ...] [--model ...]
        [--formats txt,srt_ru,...] [--recursive] [--cpu] [--config PATH]
        [--api-key KEY] [--api-model M] [--base-url URL] [--beam-size N]

重依赖均由下层模块延迟 import，本模块顶层保持轻量。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

# 支持的音频扩展名（目录扫描时按此过滤；直接指定的文件不受限制）
_AUDIO_EXTS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".mp4",
    ".webm",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ru2zh",
        description="俄语语音转写 + 俄译中：把俄语音频转写为文字并翻译成中文，"
        "输出 txt / srt / json。",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="文件或目录",
        help="一个或多个音频文件，或包含音频的目录。",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="out_dir",
        default="output",
        help="输出目录（默认 output/）。",
    )
    parser.add_argument(
        "--engine",
        choices=["nllb", "claude", "openai", "deepseek"],
        default=None,
        help="翻译引擎：nllb（本地）| claude | openai | deepseek（API）。",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Whisper 模型别名或本地路径（如 small / large-v3）。",
    )
    parser.add_argument(
        "--formats",
        default=None,
        help="输出格式，逗号分隔：txt,srt_ru,srt_zh,srt_bilingual,json。",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描输入目录中的音频。",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="强制使用 CPU（等价于 device=cpu 且不要求 GPU）。",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径（默认自动查找 config.yaml）。",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="翻译 API 密钥（也可用环境变量提供）。",
    )
    parser.add_argument(
        "--api-model",
        dest="api_model",
        default=None,
        help="翻译 API 模型名（留空用引擎默认模型）。",
    )
    parser.add_argument(
        "--base-url",
        dest="base_url",
        default=None,
        help="自定义 OpenAI 兼容端点 URL。",
    )
    parser.add_argument(
        "--beam-size",
        dest="beam_size",
        type=int,
        default=None,
        help="解码 beam 大小（越大越准但越慢）。",
    )
    parser.add_argument(
        "--nllb-dir",
        dest="nllb_dir",
        default=None,
        help="本地 NLLB 模型目录（覆盖配置中的 nllb_model_dir）。",
    )
    return parser


def _apply_args_to_cfg(cfg, args) -> None:
    """把命令行参数覆盖到 cfg 上（只覆盖显式提供的项）。"""
    if args.engine is not None:
        cfg.engine = args.engine
    if args.model is not None:
        cfg.whisper_model = args.model
    if args.formats is not None:
        cfg.output_formats = [s.strip() for s in args.formats.split(",") if s.strip()]
    if args.api_key is not None:
        cfg.api_key = args.api_key
    if args.api_model is not None:
        cfg.api_model = args.api_model
    if args.base_url is not None:
        cfg.api_base_url = args.base_url
    if args.beam_size is not None:
        cfg.beam_size = args.beam_size
    if args.nllb_dir is not None:
        cfg.nllb_model_dir = args.nllb_dir
    if args.cpu:
        cfg.device = "cpu"
        cfg.require_gpu = False


def _collect_audio_files(inputs: list[str], recursive: bool) -> list[Path]:
    """把输入（文件或目录）展开为音频文件列表（去重、按路径排序稳定）。"""
    files: list[Path] = []
    seen: set = set()

    def _add(p: Path) -> None:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            files.append(p)

    for item in inputs:
        p = Path(item)
        if p.is_file():
            _add(p)  # 直接指定的文件不做扩展名过滤
        elif p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            for f in sorted(it):
                if f.is_file() and f.suffix.lower() in _AUDIO_EXTS:
                    _add(f)
        else:
            print(f"[警告] 路径不存在，已跳过：{item}", file=sys.stderr)

    return files


def _make_progress_printer() -> Callable[[float, str], None]:
    """返回一个把进度打印在同一行（\\r 刷新）的回调。"""

    def _printer(frac: float, msg: str) -> None:
        pct = int(round(frac * 100))
        # 截断过长的阶段描述，避免残留
        if len(msg) > 44:
            msg = msg[:43] + "…"
        line = f"\r  [{pct:3d}%] {msg}"
        sys.stdout.write(line.ljust(64)[:64])
        sys.stdout.flush()

    return _printer


def main(argv: list[str] | None = None) -> int:
    from . import runtime
    from . import exporters
    from . import pipeline as pipeline_mod
    from .config import load_config

    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    _apply_args_to_cfg(cfg, args)

    # 必须在任何 huggingface_hub / 模型加载之前设置 HF 端点
    runtime.apply_hf_endpoint(cfg)

    files = _collect_audio_files(args.inputs, args.recursive)
    if not files:
        print("[错误] 未找到任何音频文件。请检查输入路径或使用 --recursive。", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    total = len(files)
    success = 0
    failed = 0
    t0 = time.time()

    print(f"共找到 {total} 个音频文件，翻译引擎：{cfg.engine}，输出目录：{out_dir}")

    for idx, audio in enumerate(files, start=1):
        print(f"\n[{idx}/{total}] 处理：{audio}")
        try:
            printer = _make_progress_printer()
            result = pipeline_mod.transcribe_and_translate(str(audio), cfg, printer)
            print()  # 进度行结束换行

            note = result.meta.get("note")
            if note:
                print(f"  提示：{note}")

            written = exporters.export_all(result, out_dir, cfg.output_formats)
            print("  已生成：")
            for p in written:
                print(f"    {p}")
            success += 1
        except Exception as e:  # noqa: BLE001 单文件失败不影响其余文件
            print()  # 断开可能残留的进度行
            print(f"  [失败] {audio}：{e}", file=sys.stderr)
            failed += 1

    elapsed = time.time() - t0
    print(
        f"\n===== 处理完成：成功 {success} / 失败 {failed}，"
        f"总耗时 {elapsed:.1f} 秒 ====="
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
