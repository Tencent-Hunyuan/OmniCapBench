# OmniCapBench：面向细粒度音视频描述的深层结构化评测框架

[📖 指标文档](docs/METRICS_zh.md) · [English](README.md)

> 论文标题：*OmniCapBench: A Deep-Structured Evaluation Framework for Fine-Grained Audio-Visual Captioning*

## 最新动态

* **[2026/09]** 🎉 **OmniCapBench** 正式开源：包含构建深层结构化评测集的标注管线，以及完整的打分工具链。

## 目录

* [简介](#简介)
* [评测单元](#评测单元)
* [指标](#指标)
* [安装](#安装)
* [快速开始](#快速开始)
* [第一步 —— 构建评测集](#第一步--构建评测集)
* [第二步 —— 评测模型](#第二步--评测模型)
* [支持的模型](#支持的模型)
* [引用](#引用)
* [许可协议](#许可协议)

## 简介

**OmniCapBench** 把音视频描述评测重构为**深层结构化诊断**任务。它不再让全局 LLM 去打分一段自由文本，而是把预测目标转换为三条原生轨道上的**原子化、可验证评测单元**：实体引用、视觉镜头、音频事件。

这解决了现有描述类基准无法回避的取舍。整段描述类协议覆盖了整个视频，但把一切压缩成一个不透明的标量，局部错误被掩盖；探针类协议能定位错误，却只覆盖稀疏的点。两者最终都把密集的整体文本交给无约束的 LLM 评判器，导致分数随评判器强弱大幅漂移。

OmniCapBench 通过把打分算子拆成两个阶段，同时保住完整覆盖与精确定位：

1. **确定性验证。** 规则检查 schema 合法性、标识符作用域、时间边界与跨轨道链接，然后按时间 IoU 对齐镜头与事件，按受约束的 LLM 匹配器对齐引用与子镜头。
2. **局部语义检查。** LLM 评判器每次只看**一对已对齐的字段**。结构问题已在上一阶段解决，它不需要解析文本、消解标识符或追踪时间。

由于每个分数都能追溯到具体单元，失败变得可诊断：未定义的标识符、时间重叠不足、声音挂到错误镜头、幻觉属性，都被分别计入，而不是被平均掉。

论文把这套框架应用在 786 个密集标注视频上（12.8 小时，20 个一级类别与 125 个二级子类别，共产出 5,818 个引用、11,419 个镜头、39,160 个子镜头、6,537 个事件）。本仓库包含的正是生成那批标注的管线与为其打分的指标，因此同一套协议可以在任意视频集合上复现。

## 评测单元

一个视频 `V` 被形式化为结构化状态 `S*(V) = (R, E, H)`，由三条相互依赖的轨道构成。这套 schema 既是管线的输出，也是指标的输入。

| 轨道                    | 编码内容                                                       | 可验证属性                     |
| ----------------------- | -------------------------------------------------------------- | ------------------------------ |
| **引用**（`R`） | 人物、物体、场景的持久身份，每个带唯一标识符与外观锚点         | 跨剪辑的身份恒常性、外观保真度 |
| **事件**（`E`） | 听觉事件与口语对话，带连续时间范围                             | 声音召回、时间边界、转写准确度 |
| **镜头**（`H`） | 视觉时间线切分为镜头与更细的**子镜头**，并回链引用与事件 | 镜头召回、边界精度、音视绑定   |

对于开放词表字段（引用与子镜头），打分是**双向**的：召回方向以 GT 为主体，惩罚遗漏；精确方向以预测为主体，惩罚幻觉。这使模型既无法用过于简短的回答刷分，也无法靠堆砌臆测细节取巧。

字段级 schema 与两种 JSONL 记录的确切格式见 **[docs/METRICS_zh.md](docs/METRICS_zh.md)**。

## 指标

规则结构指标与局部语义指标是两张独立的表，不输出聚合总分，因此每个数字都可归因到视频状态的某个具体组成部分。

| 类别 | 指标                                                                                     | 衡量内容                                                       |
| ---- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 门槛 | `SGC`                                                                                  | Schema 合规门槛 —— 输出是否完整到足以被打分                  |
| 视觉 | `RefUse`、`CCC`                                                                      | 引用在镜头/事件/子镜头中是否被正确使用，以及跨剪辑的身份恒常性 |
| 视觉 | `Ref Subject F1`、`Ref Scene F1`                                                     | 主体（人/动物/物体）与场景引用的检出                           |
| 视觉 | `Shot F1`/`tIoU`、`Subshot F1`/`tIoU`                                            | 视觉状态召回与时间边界精度                                     |
| 音频 | `Event F1`、`Event tIoU`                                                             | 声音事件召回与时间对齐                                         |
| 音视 | `Speaker F1`、`EVSA F1`                                                              | 对话说话人归属，以及声音是否绑定到同期镜头                     |
| 语义 | `Shot`、`Subject`/`Scene`/`Subshot` 的召回与精确、`Dialogue`、`Non-dialogue` | 描述保真度，仅在结构对齐单元上计算                             |

精确公式、匹配阈值与输出字段名见 **[docs/METRICS_zh.md](docs/METRICS_zh.md)**。

## 安装

需要 Python 3.8+ 与 FFmpeg。

```bash
git clone <仓库地址>
cd OmniCapBench

python -m venv .venv && source .venv/bin/activate
pip install -e .            # 本地 GPU 推理请装 ".[local]"
```

## 快速开始

无需 API key、GPU 或视频文件，即可离线验证安装：

```bash
python examples/run_metrics_demo.py
```

该脚本用内置样例预测对内置标注打分；这两个样例同时也是工具链所用两种 JSONL 格式的参考。样例预测是 GT 的副本，因此这是一次**理想预测自检**：所有指标应当落在 1.0 或其附近。

## 第一步 —— 构建评测集

标注管线把原始视频转换为上文描述的结构化状态。它按 引用 → 事件 → 镜头 的依赖顺序执行迭代式局部精修循环：**生成 → 评估与校验 → 局部精修 → 选择**。程序化校验器强制 schema 约束，精修阶段只丰富文本表述，标识符、时间边界与跨轨道链接保持不变——因此语言上的润色不会引入新的幻觉，也不会破坏已验证的拓扑结构。

输入是每个视频一条记录，只需要路径和镜头边界：

```json
[{"DID": "vid_001", "video_path": "data/videos/example.mp4", "audio_path": "", "shot_time": [[0.0, 5.0], [5.0, 10.0]]}]
```

```bash
# 单个视频，便于逐阶段检查输出
python -m omnicapbench_pipeline.run \
  --input examples/generation_input_sample.json --output output/generation --single 0

# 批量
python -m omnicapbench_pipeline.run \
  --input my_videos.json --output output/generation --workers 4
```

常用选项：`--limit N` 只处理前 N 条、`--sample N` 按时长分桶分层抽样、`--no-resume` 忽略断点、`--model` 覆盖所有阶段的模型。生成模型通过 `OMNICAP_QWEN_OMNI_BASE_URL`、`GEMINI_API_KEY` 或 `OPENROUTER_API_KEY` 配置，详见 [.env.example](.env.example)。

产出的标注集与 `examples/eval_sample.jsonl` 格式一致，可直接作为第二步的 GT 使用。

## 第二步 —— 评测模型

```bash
cp .env.example .env     # 只填你实际要用的项

# 生成结构化预测
VIDEO_ROOT=data/videos WORKERS=16 \
  bash scripts/run_inference.sh gemini_25 my_benchmark/eval.jsonl output/eval

# 打分
JUDGE_BASE_URL=https://api.openai.com/v1 JUDGE_MODEL=gpt-4o JUDGE_API_KEY=$OPENAI_API_KEY \
  bash scripts/eval_all_metrics.sh all my_benchmark/eval.jsonl output/eval/gemini_25

# 横向对比多个模型
python -m omnicapbench_eval.report_tables \
  Gemini-2.5-Pro=output/eval/gemini_25 \
  Qwen3-Omni=output/eval/qwen3_omni_instruct --output output/report
```

标注里的 `video_path` 可以是相对路径，用 `--video-root` 指定媒体根目录来解析。`video_path` 字段本身永不被改写，因为它是把预测关联回 GT 的键。

推理每个样本后都会存断点，重跑同一命令即续跑。打分模式包括 `all`、`rule-based`、`llm-only`、`convert-only`；结果落在预测目录下，包括 `metrics_summary.json`、`metrics_summary_detail.jsonl`、`llm_judge_scores.json` 与 `report_tables.md`。

任何 OpenAI 兼容端点都可作为评判器：OpenAI、OpenRouter、DashScope，或自托管 vLLM。同一端点也驱动规则指标内部的 LLM 辅助引用匹配，从而让 `Ref`、`RefUse`、`CCC` 与语义分保持一致。由于评判器只做受限的局部比较，分数在不同强度的评判器之间保持稳定。不配评判端点时规则指标仍可运行，但引用匹配会回退为标识符精确匹配。

所有脚本都支持 `--help`；`run_eval.py` 另外支持 `--dry-run` 预览、`--start`/`--end` 多机分片、`--no-resume` 重新开始。

## 支持的模型

| 模型                                                            | 接入方式 | 配置                                       |
| --------------------------------------------------------------- | -------- | ------------------------------------------ |
| Gemini 3.1-Pro / 2.5-Pro                                        | 云端     | `GEMINI_API_KEY`                         |
| Qwen3.5-Omni-Plus / Flash                                       | 云端     | `OMNICAP_EVAL_QWEN35_API_KEY` + base URL |
| GPT / Claude 等 200+ 模型                                       | 云端     | `OPENROUTER_API_KEY` + 模型名            |
| Qwen3-Omni-30B-A3B（Instruct / Thinking / Captioner）           | 本地     | vLLM ——`bash scripts/serve_vllm.sh`    |
| Qwen2.5-Omni-7B                                                 | 本地     | vLLM                                       |
| ASID-Captioner、AVoCaDO、TimeChat-Captioner、UGC-VideoCaptioner | 本地     | Transformers +`HF_TOKEN`                 |

用 `python -m omnicapbench_eval.download_models --all` 下载开源权重。

针对自由文本做过微调的描述模型对结构化输出指令的遵循很差。对这类模型，先用 `CONVERT_MODEL_KIND=<name> bash scripts/eval_all_metrics.sh convert-only ...` 把输出转换为 MTSS 单元，并把所得分数理解为「先解析后打分」基线，而非原生结构化生成的结果。

用 `pytest` 运行离线测试；它们只读本地文件，不发起任何 API 调用。

## 引用

```bibtex
@misc{omnicapbench2026,
  title  = {OmniCapBench: A Deep-Structured Evaluation Framework for Fine-Grained Audio-Visual Captioning},
  author = {OmniCapBench Team},
  year   = {2026}
}
```

## 许可协议

本仓库中的代码以 **Apache License 2.0** 发布，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

这里不分发任何视频数据或标注。如果你基于第三方视频集合构建评测集，各来源都有自己的许可与再分发条款——再分发或商用前请先核对，并把素材视作可能包含人脸、语音等可识别内容的真实世界视频。
