# OmniCapBench：面向细粒度音视频描述的深层结构化评测框架

[🤗 评测数据](https://huggingface.co/datasets/OmniCapBench/OmniCapBench) · [📖 指标文档](docs/METRICS_zh.md) · [🧭 使用手册](docs/COOKBOOK.md) · [English](README.md)

> 论文标题：*OmniCapBench: A Deep-Structured Evaluation Framework for Fine-Grained Audio-Visual Captioning*

## 最新动态

* **[2026/09]** 🎉 **OmniCapBench** 正式开源。786 个视频的评测集与完整评测工具链已在 [🤗 Hugging Face](https://huggingface.co/datasets/OmniCapBench/OmniCapBench) 公开。

## 目录

* [最新动态](#最新动态)
* [简介](#简介)
* [数据统计](#数据统计)
* [评测框架](#评测框架)
* [指标](#指标)
* [安装](#安装)
* [下载评测集](#下载评测集)
* [快速开始](#快速开始)
* [评测一个模型](#评测一个模型)
* [评测结果](#评测结果)
* [支持的模型](#支持的模型)
* [标注生成管线](#标注生成管线)
* [工程结构](#工程结构)
* [引用](#引用)
* [许可协议](#许可协议)

## 简介

**OmniCapBench** 把音视频描述评测重构为**深层结构化诊断**任务。它不再让全局 LLM 去打分一段自由文本，而是把预测目标转换为三条原生轨道上的**原子化、可验证评测单元**：实体引用、视觉镜头、音频事件。

这解决了现有描述类基准无法回避的取舍。整段描述类协议覆盖了整个视频，但把一切压缩成一个不透明的标量，局部错误被掩盖；探针类协议能定位错误，却只覆盖稀疏的点。两者最终都把密集的整体文本交给无约束的 LLM 评判器，导致分数随评判器强弱大幅漂移。

OmniCapBench 通过把打分算子拆成两个阶段，同时保住完整覆盖与精确定位：

1. **确定性验证。** 规则检查 schema 合法性、标识符作用域、时间边界与跨轨道链接，然后按时间 IoU 对齐镜头与事件，按受约束的 LLM 匹配器对齐引用与子镜头。
2. **局部语义检查。** LLM 评判器每次只看**一对已对齐的字段**，回答一个受限的比较问题。结构问题已在上一阶段解决，评判器不需要解析文本、消解标识符或追踪时间。

由于每个分数都能追溯到具体单元，失败变得可诊断：未定义的标识符、时间重叠不足、声音挂到错误镜头、幻觉属性，都被分别计入，而不是被平均成一个数字。

| | 评测单元 | 打分算子 | 覆盖度 | 定位性 | 评判稳定性 |
|---|---|---|---|---|---|
| 整段描述 | 描述对 | 全局 LLM | ✅ | ❌ | ❌ |
| 探针式 | 稀疏 QA / 完形填空 | LLM 打分器 | ❌ | ✅ | ❌ |
| 先解析后打分 | 事后解析的脚本 | 解析器 + LLM | ⚠️ | ✅ | ⚠️ |
| **OmniCapBench** | **原生原子单元** | **规则 + 受约束的局部 LLM** | ✅ | ✅ | ✅ |

## 数据统计

786 个密集标注视频，总时长 12.8 小时，覆盖 20 个一级类别与 125 个二级子类别。

| 统计项 | 单视频均值 | 总量 |
|---|---|---|
| 视频（<1 分钟） | 61.32% | 482 |
| 视频（1–3 分钟） | 29.01% | 228 |
| 视频（3–5 分钟） | 9.67% | 76 |
| 引用（References） | 7.40 | 5,818 |
| 镜头（Shots） | 14.53 | 11,419 |
| 子镜头（Subshots） | 49.82 | 39,160 |
| 事件（Events） | 8.32 | 6,537 |
| — 其中对话 | 6.83 | 5,370 |

## 评测框架

一个视频 `V` 被形式化为结构化状态 `S*(V) = (R, E, H)`，由三条相互依赖的轨道构成。

| 轨道 | 编码内容 | 可验证属性 |
|---|---|---|
| **引用**（`R`） | 人物、物体、场景的持久身份，每个带唯一标识符与外观锚点 | 跨剪辑的身份恒常性、外观保真度 |
| **事件**（`E`） | 听觉事件与口语对话，带连续时间范围 | 声音召回、时间边界、转写准确度 |
| **镜头**（`H`） | 视觉时间线切分为镜头与更细的**子镜头**，并回链引用与事件 | 镜头召回、边界精度、音视绑定 |

模型通过三项任务被评测：

1. **原生单元生成** —— 模型直接输出三条轨道构成的结构化 JSON，而非一段文字。
2. **结构验证与匹配** —— 确定性检查剔除标识符未定义、起止时间倒置、事件-镜头链接不可能成立的单元，余下的与 GT 对齐。
3. **局部语义打分** —— LLM 评判器只比较已对齐的内容字段。

对于开放词表字段（引用与子镜头），打分是**双向**的：召回方向以 GT 为主体，惩罚遗漏；精确方向以预测为主体，惩罚幻觉。这使模型既无法用过于简短的回答刷分，也无法靠堆砌臆测细节取巧。

## 指标

规则结构指标与局部语义指标是两张独立的表，不输出聚合总分，因此每个数字都可归因到视频状态的某个具体组成部分。

**规则结构指标。** `SGC` 是独立的 schema 前置门槛，不是质量分。

| 类别 | 指标 | 衡量内容 |
|---|---|---|
| 门槛 | `SGC` | Schema 合规门槛 —— 输出是否完整到足以被打分 |
| 视觉 | `RefUse` | 引用是否在镜头、事件、子镜头中被正确使用 |
| 视觉 | `CCC` | 跨镜头共指一致性 —— 跨剪辑的身份恒常性 |
| 视觉 | `Ref Subject F1` / `Ref Scene F1` | 主体（人/动物/物体）与场景引用的检出 |
| 视觉 | `Shot F1` / `Shot tIoU` | 镜头召回与边界精度 |
| 视觉 | `Subshot F1` / `Subshot tIoU` | 细粒度视觉状态召回与边界精度 |
| 音频 | `Event F1` / `Event tIoU` | 声音事件召回与时间对齐 |
| 音视 | `Speaker F1` | 对话是否归属到正确的视觉说话人 |
| 音视 | `EVSA F1` | 事件-镜头关联 —— 声音是否绑定到同期视觉镜头 |

**局部语义指标**，仅在结构上已对齐的单元上计算：

| 类别 | 指标 | 衡量内容 |
|---|---|---|
| 视觉 | `Shot` | 摄影与转场事实的保真度 |
| 视觉 | `Subject Recall` / `Subject Precision` | 主体外观描述，双向 |
| 视觉 | `Scene Recall` / `Scene Precision` | 场景描述，双向 |
| 视觉 | `Subshot Recall` / `Subshot Precision` | 细粒度视觉状态描述，双向 |
| 音频 | `Dialogue` | 基于规则的对话行准确度（归一化 WER） |
| 音频 | `Non-dialogue` | 非对话事件描述的语义质量 |

精确公式、匹配阈值与输出字段名见 **[docs/METRICS_zh.md](docs/METRICS_zh.md)**。

## 安装

需要 Python 3.8+ 与 FFmpeg。

```bash
git clone https://github.com/OmniCapBench/OmniCapBench.git
cd OmniCapBench

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

如需在本地 GPU 上推理开源权重：

```bash
pip install -e ".[local]"
pip install vllm            # 可选，用于部署大型全模态模型
```

FFmpeg 用于全部视频与音频预处理：

```bash
sudo apt-get install ffmpeg      # Debian / Ubuntu
brew install ffmpeg              # macOS
ffmpeg -version                  # 验证
```

## 下载评测集

```bash
bash scripts/download_dataset.sh data/OmniCapBench
```

该脚本从 [🤗 OmniCapBench/OmniCapBench](https://huggingface.co/datasets/OmniCapBench/OmniCapBench) 拉取约 12 GB 数据，目录结构如下：

```text
data/OmniCapBench/
├── eval.jsonl              # 786 条标注记录，字段名 video_path
└── test/
    ├── metadata.jsonl      # 同样的记录，字段名 file_name（供 Dataset Viewer 使用）
    └── videos/             # 786 个媒体文件
```

标注中的视频路径是相对路径，因此需要用 `--video-root data/OmniCapBench/test`（或 `OMNICAP_EVAL_VIDEO_ROOT`）指定媒体根目录。`video_path` 字段本身永不被改写，因为它是把预测关联回 GT 的键。

## 快速开始

无需 API key、GPU 或视频文件，即可离线验证安装：

```bash
python examples/run_metrics_demo.py
```

该脚本用内置样例预测对内置标注打分，并打印完整的规则指标表。样例预测是 GT 的副本，因此这是一次**理想预测自检** —— 所有指标应当落在 1.0 或其附近。此处引用匹配回退为标识符精确匹配；配置评判端点后才会启用 LLM 辅助匹配。

## 评测一个模型

### 1. 配置凭据

```bash
cp .env.example .env     # 只填你实际要用的项
```

完整变量清单见 [.env.example](.env.example)。

### 2. 生成结构化预测

```bash
VIDEO_ROOT=data/OmniCapBench/test WORKERS=16 \
  bash scripts/run_inference.sh gemini_25 data/OmniCapBench/eval.jsonl output/eval
```

输出写入 `output/eval/gemini_25/`：

```text
results.jsonl      成功的预测
errors.jsonl       失败样本及 traceback
progress.json      断点记录；重跑同一命令即续跑
```

### 3. 对预测打分

```bash
JUDGE_BASE_URL=https://api.openai.com/v1 \
JUDGE_MODEL=gpt-4o \
JUDGE_API_KEY=$OPENAI_API_KEY \
  bash scripts/eval_all_metrics.sh all data/OmniCapBench/eval.jsonl output/eval/gemini_25
```

模式包括 `all`、`rule-based`、`llm-only`、`convert-only`。同一评判端点也驱动规则指标内部的 LLM 辅助引用匹配，从而让 `Ref`、`RefUse`、`CCC` 与语义分保持一致。

结果落在预测目录下：

```text
metrics_summary.json          规则结构指标
metrics_summary_detail.jsonl  逐视频明细
llm_judge_scores.json         局部语义指标
report_tables.md              渲染好的两张结果表
```

### 4. 横向对比多个模型

```bash
python -m omnicapbench_eval.report_tables \
  Gemini-2.5-Pro=output/eval/gemini_25 \
  Qwen3-Omni=output/eval/qwen3_omni_instruct \
  --output output/report
```

任何 OpenAI 兼容端点都可作为评判器：OpenAI、OpenRouter、DashScope，或自托管 vLLM。由于评判器只做受限的局部比较，分数在不同强度的评判器之间保持稳定；评判敏感性验证步骤见 [docs/COOKBOOK.md](docs/COOKBOOK.md)。

## 评测结果

规则结构评测，按视频宏平均（%）：

| 模型 | SGC | RefUse | CCC | Ref Subject F1 | Ref Scene F1 | Shot F1 | Shot tIoU | Subshot F1 | Subshot tIoU | Event F1 | Event tIoU | Speaker F1 | EVSA F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemini 3.1-Pro | **97.36** | 70.39 | 34.43 | 78.60 | 82.68 | **76.24** | 70.74 | 65.27 | 48.76 | **63.32** | **71.92** | 91.18 | **51.46** |
| Gemini 2.5-Pro | 96.73 | **70.75** | **37.81** | **80.70** | **83.65** | 70.93 | **72.58** | **66.69** | **49.18** | 58.52 | 69.35 | 89.72 | 43.74 |
| Qwen3.5-Omni-Plus | 96.09 | 66.92 | 29.55 | 73.57 | 79.97 | 71.25 | 72.35 | 65.69 | 48.29 | 53.92 | 67.77 | **91.20** | 40.20 |
| Qwen3.5-Omni-Flash | 95.58 | 63.10 | 24.31 | 72.27 | 66.07 | 65.65 | 68.99 | 58.94 | 46.52 | 50.68 | 69.34 | 89.91 | 35.58 |
| Qwen3-Omni-Instruct | 89.02 | 40.96 | 9.85 | 54.84 | 64.55 | 49.98 | 67.92 | 29.51 | 47.95 | 41.70 | 40.99 | 88.35 | 13.33 |
| Qwen3-Omni-Captioner | 90.77 | 44.96 | 11.51 | 54.36 | 57.08 | 52.29 | 70.80 | 39.51 | **49.69** | 41.31 | 52.60 | 88.68 | 17.42 |

局部语义评测，仅在结构对齐单元上计算（%）：

| 模型 | Shot | Subject Recall | Subject Precision | Scene Recall | Scene Precision | Subshot Recall | Subshot Precision | Dialogue | Non-dialogue |
|---|---|---|---|---|---|---|---|---|---|
| Gemini 3.1-Pro | **59.05** | 43.78 | **67.07** | 51.78 | **75.62** | **53.29** | 70.92 | 91.18 | **41.49** |
| Gemini 2.5-Pro | 55.58 | **51.53** | 57.08 | **56.80** | 65.12 | 53.05 | 71.21 | 89.72 | 37.38 |
| Qwen3.5-Omni-Plus | 54.17 | 45.41 | 62.84 | 50.75 | 69.80 | 50.22 | 70.66 | **91.20** | 36.82 |
| Qwen3.5-Omni-Flash | 47.66 | 43.87 | 65.59 | 44.41 | 71.26 | 44.08 | 67.27 | 89.91 | 28.12 |
| Qwen3-Omni-Instruct | 34.63 | 40.31 | 62.02 | 36.07 | 70.73 | 29.42 | 57.84 | 88.35 | 20.51 |
| Qwen3-Omni-Captioner | 36.08 | 42.00 | 50.96 | 38.07 | 54.53 | 34.01 | 46.05 | 88.68 | 19.65 |

三个结论尤为突出。Schema 遵循基本已被解决（前沿模型 `SGC` 均超过 95%），但**身份恒常性并未解决**：最强模型的 `CCC` 仅 37.81%，开源模型接近 10%——这是只检查实体是否被提及过一次的指标完全看不到的缺陷。模型的语音转写很好（`Dialogue` 普遍高于 88%），却**无法把声音绑定到其视觉来源**：Gemini 3.1-Pro 从 63.32% 的 `Event F1` 掉到 51.46% 的 `EVSA F1`。最后，把语义打分限制在对齐单元上，暴露出流畅文本通常会掩盖的**精确率-召回率取舍**：Gemini 3.1-Pro 偏保守、以省略换安全（主体精确率 67.07%，召回率仅 43.78%），而较弱的模型靠幻觉属性抬高召回。

## 支持的模型

| 模型 | 接入方式 | 配置 |
|---|---|---|
| Gemini 3.1-Pro / 2.5-Pro | 云端 | `GEMINI_API_KEY` |
| Qwen3.5-Omni-Plus / Flash | 云端 | `OMNICAP_EVAL_QWEN35_API_KEY` + base URL |
| GPT / Claude 等 200+ 模型 | 云端 | `OPENROUTER_API_KEY` + 模型名 |
| Qwen3-Omni-30B-A3B（Instruct / Thinking / Captioner） | 本地 | vLLM —— `bash scripts/serve_vllm.sh` |
| Qwen2.5-Omni-7B | 本地 | vLLM |
| ASID-Captioner、AVoCaDO、TimeChat-Captioner、UGC-VideoCaptioner | 本地 | Transformers + `HF_TOKEN` |

用 `python -m omnicapbench_eval.download_models --all` 下载开源权重。

针对自由文本做过微调的描述模型对结构化输出指令的遵循很差。对这类模型，先用 `CONVERT_MODEL_KIND=<name> bash scripts/eval_all_metrics.sh convert-only ...` 把输出转换为 MTSS 单元，并把所得分数理解为「先解析后打分」基线，而非原生结构化生成的结果。

## 标注生成管线

本仓库同时包含构建评测标注所用的管线，因此结构化轨道可以被重新生成或扩展到新视频。它按 引用 → 事件 → 镜头 的依赖顺序执行迭代式局部精修循环：**生成 → 评估与校验 → 局部精修 → 选择**。程序化校验器强制 schema 约束，精修阶段只丰富文本表述，标识符、时间边界与跨轨道链接保持不变。

```bash
python -m omnicapbench_pipeline.run \
  --input examples/generation_input_sample.json \
  --output output/generation \
  --workers 4
```

细节与模型接线见 [docs/COOKBOOK.md](docs/COOKBOOK.md)。

## 工程结构

```text
OmniCapBench/
├── omnicapbench_eval/            # 评测工具链
│   ├── run_eval.py               # 在评测集上做结构化推理
│   ├── inference_engine.py       # 按模型族的视频采样与提示构造
│   ├── mtss_metrics.py           # 规则结构指标
│   ├── mtss_llm_judge.py         # 局部语义打分
│   ├── report_tables.py          # 渲染两张结果表
│   ├── mtss_transformer_convert.py  # 自由文本描述 → MTSS 单元
│   ├── native_vs_posthoc.py      # 原生生成 vs. 事后解析的对照实验
│   ├── download_models.py        # 开源权重下载器
│   └── mtss_generate_prompts/    # 结构化生成提示词
│
├── omnicapbench_pipeline/        # 标注构建管线
│   ├── run.py                    # CLI 入口
│   ├── pipeline.py               # 局部精修循环
│   ├── stages/                   # 生成 / 评估 / 精修 / 选择
│   ├── validators.py             # schema 与跨轨道链接校验
│   └── prompts/                  # 各阶段提示词构造
│
├── scripts/                      # Shell 入口脚本
├── examples/                     # 离线演示、样例标注与预测
├── docs/                         # 指标文档与使用手册
└── tests/                        # 离线测试
```

用 `pytest` 运行离线测试；它们只读本地文件，不发起任何 API 调用。

## 引用

```bibtex
@misc{omnicapbench2026,
  title        = {OmniCapBench: A Deep-Structured Evaluation Framework for Fine-Grained Audio-Visual Captioning},
  author       = {OmniCapBench Team},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/OmniCapBench/OmniCapBench}}
}
```

## 许可协议

本仓库中的评测代码以 **MIT License** 发布，见 [LICENSE](LICENSE)。

评测视频单独分发，聚合自多个上游数据集，各自有不同的许可与再分发条款。再分发或商用前请先核对来源数据集的条款，并把数据视作可能包含人脸、语音等可识别内容的真实世界视频。
