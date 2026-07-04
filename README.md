# ru2zh —— 俄语录音转写 + 俄译中

把俄语录音一键转成**俄语文字**和**中文翻译**的桌面小工具。面向不会编程的普通用户，全程双击 `.bat` 即可使用。

---

## 目录

- [1. 软件简介](#1-软件简介)
- [2. 系统要求](#2-系统要求)
- [3. 安装](#3-安装)
- [4. 使用](#4-使用)
- [5. 翻译引擎切换](#5-翻译引擎切换)
- [6. 配置说明（config.yaml）](#6-配置说明configyaml)
- [7. 显存说明](#7-显存说明)
- [8. 常见问题 FAQ](#8-常见问题-faq)
- [9. 首次运行检查清单](#9-首次运行检查清单)
- [10. 目录结构与卸载](#10-目录结构与卸载)

---

## 1. 软件简介

ru2zh 做两件事：

1. **语音识别（俄语录音 → 俄语文字）**：使用 `faster-whisper` 的 `large-v3` 模型，对口语、带口音、语速快的录音也有不错的效果。
2. **翻译（俄语文字 → 中文）**：默认使用**本地离线**的 NLLB 模型，完全免费、不联网、不上传任何内容；也可以切换到 **Claude / OpenAI / DeepSeek** 等在线大模型，翻译质量更高（需自备 API 密钥，按量付费）。

两种使用方式：

| 方式 | 启动 | 适合 |
| --- | --- | --- |
| **网页界面** | 双击 `start_webui.bat` | 上传单个文件、麦克风录音、即时查看俄中对照、下载导出 |
| **命令行批量** | 双击 `transcribe.bat` 或拖入文件 | 一次处理很多文件、整个文件夹批量转写 |

**输出格式**（可自选）：

| 格式 | 文件后缀 | 说明 |
| --- | --- | --- |
| `txt` | `.txt` | 带时间轴的俄中对照纯文本 |
| `srt_ru` | `.ru.srt` | 俄文字幕 |
| `srt_zh` | `.zh.srt` | 中文字幕 |
| `srt_bilingual` | `.bi.srt` | 俄中双语字幕（上俄下中） |
| `json` | `.json` | 结构化数据，方便二次开发 |

---

## 2. 系统要求

| 项目 | 要求 |
| --- | --- |
| 操作系统 | Windows 10 / 11，64 位 |
| 显卡 | **NVIDIA 独立显卡**（本软件默认强制使用 GPU；实测 RTX 3070 8GB 显存流畅运行） |
| 显卡驱动 | 最新版 NVIDIA 驱动（[下载地址](https://www.nvidia.cn/geforce/drivers/)） |
| Python | 3.10 / 3.11 / 3.12（推荐 3.12） |
| 磁盘空间 | ≥ 15 GB（模型约 8GB + 依赖库 + 输出文件） |
| 网络 | 首次安装需要联网下载约 8GB 模型；之后可离线使用（在线翻译引擎除外） |

> 没有 NVIDIA 显卡也能跑，但速度很慢，见 [FAQ 第 ⑥ 条](#8-常见问题-faq)。

---

## 3. 安装

只需三步。

### 第一步：安装 Python

1. 打开官网下载页：<https://www.python.org/downloads/windows/>
2. 下载 Python 3.10 ~ 3.12 的 64 位安装包并运行。
3. **安装第一个界面务必勾选 `Add python.exe to PATH`**（把 Python 加入 PATH），否则后面找不到 Python。
4. 点 `Install Now` 完成安装。

### 第二步：双击 `install.bat`

在软件文件夹里双击 `install.bat`，它会自动完成：

```
查找 Python  ->  创建虚拟环境 .venv  ->  升级 pip  ->  安装运行依赖
->  安装 CUDA 运行库（cuDNN 约 700MB）  ->  环境诊断  ->  下载模型（约 8GB）
```

安装过程全自动，中途请勿关闭窗口。看到 `安装完成！` 字样即成功。

> **关于 SmartScreen 拦截**：首次运行 `.bat` 时，Windows 可能弹出蓝色的
> "Windows 已保护你的电脑" 提示。点击提示里的 **更多信息**，再点 **仍要运行** 即可。
> 这是所有本地脚本的通用现象，本软件不含任何联网上传行为。

### 第三步：等待完成

- 模型下载较慢（约 8GB），**支持断点续传**；万一中途断网，重新双击 `install.bat` 会接着下载。
- 若无法访问 huggingface.co，脚本会**自动切换到国内镜像** `hf-mirror.com`。
- pip 安装慢时，脚本会**自动改用清华镜像**重试一次。

安装完成后，双击 `start_webui.bat` 即可开始使用。

---

## 4. 使用

### 4.1 网页界面（推荐新手）

1. 双击 `start_webui.bat`。
2. 稍等片刻，浏览器会**自动打开** <http://127.0.0.1:7860>（没自动弹出就手动在浏览器地址栏输入这个网址）。
3. 在页面里 **上传录音文件** 或 **用麦克风录音**。
4. 等待处理，得到**俄中对照**结果。
5. 点击下载按钮，导出 txt / SRT 字幕 / JSON。
6. 用完关闭那个黑色命令行窗口即可停止服务。

### 4.2 命令行批量处理

双击 `transcribe.bat`（不带参数）会显示用法说明。常用命令示例（也可以直接把文件或文件夹**拖到 `transcribe.bat` 图标上**运行）：

```bat
:: 转写单个文件
transcribe.bat 录音.mp3

:: 转写整个文件夹（含子文件夹）
transcribe.bat D:\录音文件夹 --recursive

:: 指定翻译引擎（本次改用 Claude 在线翻译）
transcribe.bat 录音.mp3 --engine claude

:: 只导出纯文本和双语字幕
transcribe.bat 录音.mp3 --formats txt,srt_bilingual

:: 没有 GPU 时强制用 CPU 运行（很慢）
transcribe.bat 录音.mp3 --cpu
```

处理结果默认保存在 `output` 文件夹中。

---

## 5. 翻译引擎切换

ru2zh 支持四种翻译引擎，可随时切换：

| 引擎（`engine` 值） | 类型 | 是否收费 | 特点 |
| --- | --- | --- | --- |
| `nllb` | 本地离线 | 免费 | 默认引擎，不联网、隐私安全；质量够用，专有名词偶有偏差 |
| `claude` | 在线 API | 按量付费 | 翻译流畅、上下文理解好，适合正式材料 |
| `openai` | 在线 API | 按量付费 | 综合质量高，生态成熟 |
| `deepseek` | 在线 API | 按量付费 | 中文表现好，价格通常更便宜 |

### 5.1 怎么切换

- **网页界面**：直接在页面的引擎下拉框里选择。
- **命令行**：加参数 `--engine claude`（临时生效一次）。
- **永久默认**：修改 `config.yaml` 里的 `engine:` 这一行，例如改成 `engine: deepseek`。

### 5.2 在线引擎需要 API 密钥

三种在线引擎各自读取一个环境变量：

| 引擎 | 环境变量名 | 密钥申请入口 |
| --- | --- | --- |
| `claude` | `ANTHROPIC_API_KEY` | <https://console.anthropic.com/> |
| `openai` | `OPENAI_API_KEY` | <https://platform.openai.com/> |
| `deepseek` | `DEEPSEEK_API_KEY` | <https://platform.deepseek.com/> |

**在 Windows 设置环境变量的步骤：**

1. 按 `Win` 键，搜索 **"编辑系统环境变量"** 并打开。
2. 点右下角 **环境变量(N)...** 按钮。
3. 在上方 **用户变量** 区域点 **新建(N)...**。
4. 变量名填对应的名字（如 `ANTHROPIC_API_KEY`），变量值填你的密钥，确定。
5. **关闭并重新打开** `start_webui.bat` / `transcribe.bat` 才会生效。

> 也可以不设环境变量，直接在网页界面里粘贴 API 密钥使用。

### 5.3 大致费用

在线 API 按用量（token）计费。粗略估算，**转写并翻译一小时录音，大约花费几美分到几十美分**，具体取决于所选引擎、模型和录音内容长度。本地 `nllb` 引擎完全免费。

---

## 6. 配置说明（config.yaml）

软件根目录下的 `config.yaml` 保存所有默认设置，用记事本即可编辑（保存为 UTF-8）。每一项也可以用环境变量 `RU2ZH_<键名大写>` 覆盖。

| 配置键 | 默认值 | 说明 |
| --- | --- | --- |
| `whisper_model` | `large-v3` | 语音识别模型；可填尺寸别名（如 `small`）或本地模型目录路径 |
| `device` | `auto` | 计算设备：`auto`（自动）/ `cuda`（GPU）/ `cpu` |
| `compute_type` | `auto` | 计算精度：`auto` / `float16` / `int8_float16` / `int8` |
| `beam_size` | `5` | 解码 beam 大小，越大越准但越慢 |
| `vad_min_silence_ms` | `500` | 语音分段的最短静音时长（毫秒） |
| `initial_prompt` | `null` | 初始提示词，可给识别提供专业术语上下文；留空写 `null` |
| `require_gpu` | `null` | 是否强制用 GPU：`null`=自动（Windows 上为强制）/ `true` / `false` |
| `engine` | `nllb` | 翻译引擎：`nllb` / `claude` / `openai` / `deepseek` |
| `nllb_model_dir` | `models/nllb-200-distilled-1.3B-ct2` | 本地 NLLB 模型目录 |
| `api_model` | `null` | 在线引擎的模型名；`null`=用该引擎默认模型 |
| `api_base_url` | `null` | 自定义 OpenAI 兼容端点；一般留空 |
| `api_key` | `null` | API 密钥；一般留空（改用环境变量），也可直接填 |
| `models_dir` | `models` | 模型总目录 |
| `hf_endpoint` | `null` | HuggingFace 端点；国内可填 `https://hf-mirror.com` |
| `output_formats` | `txt, srt_ru, srt_zh, srt_bilingual, json` | 默认导出的格式列表 |

> 值写 `null` 表示"留空"。修改后保存即可，下次启动生效。

---

## 7. 显存说明

本软件默认针对 **8GB 显存** 做了保守设置，保证稳定不爆显存：

| 场景 | Whisper 精度 | 翻译 | 大致显存占用 | 说明 |
| --- | --- | --- | --- | --- |
| 默认（本地翻译） | `int8_float16` | NLLB `int8_float16` | ≈ 5 GB | **安全推荐**，Whisper 与 NLLB 同时驻留显存 |
| 在线引擎模式 | `int8_float16` | 在线 API（不占显存） | 宽裕 | 翻译在云端完成，显存非常充裕 |
| 想榨精度（手动） | `float16` | NLLB 共存 | 偏紧 | 精度略高但差别极小；与 NLLB 共存时显存吃紧，可能不稳 |

- `compute_type: auto`（默认）在 GPU 上统一选 `int8_float16`：与 `float16` 的识别精度差别可以忽略，却更省显存，且在界面上切换翻译引擎时无需重新加载模型。
- 想手动提升识别精度：把 `config.yaml` 的 `compute_type` 改成 `float16`（本地翻译模式下请留意显存）。
- 显存更小的显卡：改成 `int8_float16` 或 `int8`，见 [FAQ 第 ⑤ 条](#8-常见问题-faq)。

---

## 8. 常见问题 FAQ

### ① 提示"未检测到 GPU / CUDA"

按顺序排查：

1. 确认已安装**最新 NVIDIA 显卡驱动**。
2. 重新运行一次 `install.bat`（确保 `nvidia-cublas-cu12` 和 `nvidia-cudnn-cu12` 装好）。
3. 运行诊断：`.venv\Scripts\python scripts\check_env.py`，看是否显示"检测到 N 个 CUDA 设备"。
4. **最后手段（备用方案）**：到
   <https://github.com/Purfview/whisper-standalone-win/releases>
   下载 `cuBLAS.and.cuDNN_win_v2` 压缩包，解压出里面的 `.dll` 文件，放进本项目的
   `.venv\Scripts\` 目录（或把该目录加入系统 PATH），再重试。

### ② 模型下载失败 / 很慢

- 脚本会**自动切换**到国内镜像 `hf-mirror.com`。若仍慢，可手动设置后重跑：
  在命令行执行 `set HF_ENDPOINT=https://hf-mirror.com`，再运行 `install.bat`；
  或在 `config.yaml` 里填 `hf_endpoint: https://hf-mirror.com`。
- 模型下载**支持断点续传**，断了重跑 `install.bat` 即可接着下。
- **pip 安装慢**：脚本已内置清华镜像自动回退。手动重试示例：
  ```bat
  .venv\Scripts\python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

### ③ 控制台中文乱码

`.bat` 文件已内置 `chcp 65001`（UTF-8）。个别老终端仍可能显示乱码，**不影响任何功能**，可忽略。

### ④ 麦克风不可用

浏览器只在 **`http://127.0.0.1`（本机地址）** 下才允许网页访问麦克风。
请**用本机浏览器**打开 <http://127.0.0.1:7860>，**不要**用局域网 IP（如 `192.168.x.x`）访问，否则麦克风会被浏览器禁用。

### ⑤ 显存不足（OOM，报显存相关错误）

- 把 `config.yaml` 的 `compute_type` 改成 `int8_float16` 或更省的 `int8`。
- 关闭其他占用显存的程序（浏览器多标签、游戏、其他 AI 软件等）。
- 或改用在线翻译引擎（`engine: claude/openai/deepseek`），让 Whisper 独占显存。

### ⑥ 能不能用 CPU 运行（没有独立显卡）？

可以，但**速度很慢**（可能是 GPU 的几十倍时间）。两种方式：

- 命令行加参数：`transcribe.bat 录音.mp3 --cpu`
- 或在 `config.yaml` 里设 `require_gpu: false`（同时可设 `device: cpu`）。

### ⑦ 杀毒软件误报 `.bat`

部分杀毒软件会对陌生 `.bat` 脚本敏感。请把本软件文件夹**添加到杀毒软件的信任 / 白名单**。本软件全部为本地脚本，无联网上传行为。

---

## 9. 首次运行检查清单

按下面几步确认一切正常：

- [ ] `install.bat` 运行结束，最后显示 `安装完成！`，且中途**没有红色报错**。
- [ ] `check_env.py` 输出里显示 **"检测到 N 个 CUDA 设备"**（N ≥ 1）。
- [ ] 双击 `start_webui.bat` 后，**浏览器能打开** <http://127.0.0.1:7860> 页面。
- [ ] 用**一段短录音**（十几秒即可）试转写，能得到俄文和中文结果。
- [ ] 转写时打开**任务管理器 → 性能 → GPU**，能看到 **GPU 占用上升**。

以上全部通过，说明软件已就绪。

---

## 10. 目录结构与卸载

### 目录结构

```
ru2zh/
├─ install.bat              一键安装（先运行这个）
├─ start_webui.bat          启动网页界面
├─ transcribe.bat           命令行批量转写
├─ config.yaml              配置文件（可用记事本编辑）
├─ requirements.txt         通用运行依赖清单
├─ requirements-cuda-win.txt  Windows CUDA 运行库清单
├─ README.md               本说明文件
├─ scripts/
│  ├─ download_models.py    模型下载脚本
│  └─ check_env.py          环境诊断脚本
├─ src/ru2zh/               程序源代码
├─ models/                  下载的模型（约 8GB，安装后生成）
├─ output/                  转写结果输出目录
└─ .venv/                   虚拟环境（安装后生成）
```

### 卸载

本软件**绿色免安装**，不写注册表、不留系统残留。卸载只需：

1. 关闭所有相关的命令行 / 浏览器窗口。
2. **直接删除整个软件文件夹**即可（模型都在文件夹内的 `models/` 里，一并删除）。

如果之前设置过 API 密钥的环境变量，想彻底清理，可到 [第 5.2 节](#52-在线引擎需要-api-密钥) 说的"环境变量"界面里，把 `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` 删掉。

---

> 小提示：`.bat` 文件均以 UTF-8 编码保存。若个别系统上 `echo` 出来的中文出现乱码，仅是显示问题，**不影响软件功能**。祝使用愉快！
