"""Gradio 6 网页界面：上传/麦克风录音 → 俄语转写 + 中文翻译 → 在线预览与导出下载。

本模块是重界面模块（顶层 import gradio），但 ru2zh 的其他模块都不会 import 它，
因此不会把 gradio 这类重依赖带进命令行等轻量路径。

Gradio 6 注意事项（与 5.x 不同）：
  * theme / css 等应用级参数从 Blocks() 移到了 launch()；本界面保持默认主题，故都不传。
  * launch() 不再接受 show_api 参数；改用事件级 api_visibility 控制 API 文档可见性
    （"undocumented" = 隐藏文档页但 gradio_client 仍可调用，等价于旧 show_api=False 的意图）。
"""

from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path

import gradio as gr

from .config import AppConfig, load_config
from .exporters import export_all
from .pipeline import transcribe_and_translate

# 翻译引擎下拉：显示中文标签，值为内部引擎名
_ENGINE_CHOICES = [
    ("本地 NLLB（离线免费）", "nllb"),
    ("Claude API", "claude"),
    ("OpenAI API", "openai"),
    ("DeepSeek API", "deepseek"),
]

# 高级选项里 whisper 模型候选（允许自定义值）
_WHISPER_CHOICES = ["large-v3", "medium", "small", "base"]

# 结果表头
_TABLE_HEADERS = ["时间", "俄语原文", "中文翻译"]


def _mmss(seconds: float) -> str:
    """把秒数格式化为 "MM:SS"（分钟位可超过 59，用于结果表时间列）。"""
    if seconds is None or seconds < 0:
        seconds = 0
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def _stripped(value) -> str | None:
    """把界面文本框的值规整为 None（空串/None）或去空白后的字符串。"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_app(cfg: AppConfig) -> gr.Blocks:
    """根据全局配置构建 Gradio 界面（不启动服务）。返回 gr.Blocks。

    界面上的选项只影响本次处理：处理函数会用 dataclasses.replace 复制一份 cfg
    并叠加界面覆盖，不会修改传入的全局 cfg。
    """

    def process(
        audio_path,
        engine,
        api_key,
        whisper_model,
        api_model,
        base_url,
        beam_size,
        progress=gr.Progress(),
    ):
        """处理一次转写 + 翻译，返回 (状态Markdown, 结果表行, 导出文件列表)。

        任何可预期的错误都转成状态区的中文提示，绝不让异常冒泡炸掉界面。
        """
        # 未选择音频：友好提示，不抛异常
        if not audio_path:
            return (
                "⚠️ 请先上传俄语音频文件，或用麦克风录音后再点击“开始转写并翻译”。",
                [],
                None,
            )

        # 复制全局配置并叠加界面覆盖（空字符串视为不覆盖）
        overrides: dict = {"engine": engine}
        wm = _stripped(whisper_model)
        if wm is not None:
            overrides["whisper_model"] = wm
        key = _stripped(api_key)
        if key is not None:
            overrides["api_key"] = key
        model = _stripped(api_model)
        if model is not None:
            overrides["api_model"] = model
        url = _stripped(base_url)
        if url is not None:
            overrides["api_base_url"] = url
        if beam_size is not None:
            try:
                overrides["beam_size"] = int(beam_size)
            except (TypeError, ValueError):
                pass
        run_cfg = dataclasses.replace(cfg, **overrides)

        # 把 pipeline 的 (比例, 描述) 进度回调适配到 Gradio 进度条
        def progress_cb(frac: float, desc: str) -> None:
            progress(frac, desc=desc)

        # 真正跑流水线；可预期错误统一转成状态区提示
        try:
            result = transcribe_and_translate(str(audio_path), run_cfg, progress_cb)
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            return "❌ " + str(e), [], None

        # 未检测到语音：明确提示，清空表格与文件
        if not result.segments:
            note = result.meta.get("note") or "未检测到语音"
            return f"⚠️ {note}，请确认音频中包含清晰的俄语人声。", [], None

        # 导出到 output/webui/<音频名>_<时间戳>/
        audio_name = Path(str(audio_path)).stem or "audio"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path("output") / "webui" / f"{audio_name}_{timestamp}"
        try:
            written = export_all(result, out_dir, run_cfg.output_formats)
        except OSError as e:
            # 导出失败不应吞掉已得到的结果：给出提示但仍展示表格
            written = []
            export_note = f"（导出文件时出错：{e}）"
        else:
            export_note = ""

        # 状态区：设备 / 精度 / 引擎 / 模型 / 耗时 / 段数
        meta = result.meta
        try:
            elapsed = f"{float(meta.get('elapsed', 0.0)):.1f} 秒"
        except (TypeError, ValueError):
            elapsed = str(meta.get("elapsed"))
        status = "\n".join(
            [
                "✅ 处理完成" + export_note,
                "",
                f"- 计算设备：{meta.get('device')}",
                f"- 计算精度：{meta.get('compute_type')}",
                f"- 翻译引擎：{meta.get('engine')}",
                f"- Whisper 模型：{meta.get('whisper_model')}",
                f"- 音频时长：{_mmss(meta.get('duration'))}",
                f"- 处理耗时：{elapsed}",
                f"- 段数：{meta.get('segment_count')}",
            ]
        )

        # 结果表：时间列 "MM:SS → MM:SS"
        rows = [
            [f"{_mmss(seg.start)} → {_mmss(seg.end)}", seg.ru, seg.zh]
            for seg in result.segments
        ]
        files = [str(p) for p in written] or None
        return status, rows, files

    with gr.Blocks(title="俄语语音转写与翻译") as app:
        # ---------- 标题区 ----------
        gr.Markdown(
            "# 俄语语音转写与翻译\n"
            "上传俄语音频或用麦克风录音，自动转写为俄语文字并翻译成中文，"
            "支持在线预览与导出 txt / SRT 字幕 / JSON。"
        )

        with gr.Row():
            # ---------- 左列：输入 ----------
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="上传俄语音频 / 麦克风录音",
                )
                engine_dd = gr.Dropdown(
                    choices=_ENGINE_CHOICES,
                    value=cfg.engine,
                    label="翻译引擎",
                )
                api_key_tb = gr.Textbox(
                    type="password",
                    label="API 密钥（在线引擎需要；留空则读环境变量）",
                )
                with gr.Accordion("高级选项", open=False):
                    whisper_dd = gr.Dropdown(
                        choices=_WHISPER_CHOICES,
                        value=cfg.whisper_model,
                        allow_custom_value=True,
                        label="Whisper 语音识别模型",
                    )
                    api_model_tb = gr.Textbox(
                        label="API 模型名（在线引擎，可选）",
                        placeholder="留空用默认：claude-opus-4-8 / gpt-4o / deepseek-chat",
                    )
                    base_url_tb = gr.Textbox(
                        label="自定义 API 端点（可选）",
                        placeholder="OpenAI 兼容端点，如 https://api.deepseek.com",
                    )
                    beam_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=cfg.beam_size,
                        label="解码 beam 大小（越大越准但越慢）",
                    )
                run_btn = gr.Button("开始转写并翻译", variant="primary")

            # ---------- 右列：输出 ----------
            with gr.Column(scale=1):
                status_md = gr.Markdown(
                    "填写左侧选项后，点击“开始转写并翻译”。",
                    label="状态",
                )
                result_df = gr.Dataframe(
                    headers=_TABLE_HEADERS,
                    datatype=["str", "str", "str"],
                    column_count=(3, "fixed"),
                    type="array",
                    wrap=True,
                    label="俄中对照结果",
                )
                files_out = gr.Files(label="下载导出文件")

        # 绑定按钮事件；api_name 便于 gradio_client 调用，
        # api_visibility="undocumented" 隐藏 API 文档页但仍允许客户端调用（Gradio 6 无 show_api）
        run_btn.click(
            fn=process,
            inputs=[
                audio_in,
                engine_dd,
                api_key_tb,
                whisper_dd,
                api_model_tb,
                base_url_tb,
                beam_slider,
            ],
            outputs=[status_md, result_df, files_out],
            api_name="process",
            api_visibility="undocumented",
        )

    return app


def main() -> None:
    """加载配置、构建界面并启动本地服务（自动打开浏览器）。"""
    cfg = load_config()

    # 必须在任何 huggingface_hub / 模型加载之前设置 HF 端点
    from . import runtime

    runtime.apply_hf_endpoint(cfg)

    app = build_app(cfg)
    port = int(os.environ.get("RU2ZH_PORT", "7860"))

    # 启动前打印中文横幅（地址 + 退出方式）
    banner = (
        "\n"
        "============================================================\n"
        "  俄语语音转写与翻译  ru2zh —— 网页界面正在启动……\n"
        f"  请在本机浏览器打开： http://127.0.0.1:{port}\n"
        "  （麦克风录音必须用本机 127.0.0.1 地址访问，勿用局域网 IP）\n"
        "  停止服务：在本窗口按 Ctrl+C，或直接关闭窗口\n"
        "============================================================\n"
    )
    print(banner, flush=True)

    # Gradio 6：launch() 不接受 show_api，故不传；show_api 的“隐藏文档”意图由
    # 事件级 api_visibility="undocumented" 承担（不影响 gradio_client 调用）。
    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=True,
        quiet=False,
    )


if __name__ == "__main__":
    main()
