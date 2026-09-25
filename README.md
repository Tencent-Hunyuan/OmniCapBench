# OmniCapBench: A Deep-Structured Evaluation Framework for Fine-Grained Audio-Visual Captioning

[🤗 Benchmark Data](https://huggingface.co/datasets/OmniCapBench/OmniCapBench) · [📖 Metrics Reference](docs/METRICS.md) · [🧭 Cookbook](docs/COOKBOOK.md) · [中文说明](README_zh.md)

## News

* **[2026/09]** 🎉 **OmniCapBench** is now open-source. The 786-video benchmark and the full evaluation toolkit are publicly available on [🤗 Hugging Face](https://huggingface.co/datasets/OmniCapBench/OmniCapBench).

## Contents

* [News](#news)
* [Introduction](#introduction)
* [Benchmark Statistics](#benchmark-statistics)
* [Evaluation Framework](#evaluation-framework)
* [Metrics](#metrics)
* [Installation](#installation)
* [Download the Benchmark](#download-the-benchmark)
* [Quick Start](#quick-start)
* [Evaluating a Model](#evaluating-a-model)
* [Results](#results)
* [Supported Models](#supported-models)
* [Caption Generation Pipeline](#caption-generation-pipeline)
* [Project Structure](#project-structure)
* [Citation](#citation)
* [License](#license)

## Introduction

**OmniCapBench** reframes audio–visual caption evaluation as a *deep-structured diagnostic* task. Instead of scoring a free-form paragraph with a global LLM judge, it shifts the prediction target to sets of **atomic, verifiable evaluation units** across three native tracks — entity references, visual shots, and audio events.

This resolves a trade-off that existing caption benchmarks cannot escape. Whole-caption protocols cover the whole video but compress everything into an opaque scalar that hides local errors. Probe-based protocols localize errors but cover only sparse points. Both then hand dense, monolithic text to an unconstrained LLM judge, which makes scores swing with judge strength.

OmniCapBench keeps full coverage *and* exact localization by decoupling the scoring operator into two stages:

1. **Deterministic verification.** Rules check schema validity, identifier scoping, time bounds, and cross-track links, then align predicted units to the ground truth by temporal IoU (shots, events) or a bounded LLM matcher (references, subshots).
2. **Localized semantic checking.** An LLM judge sees only *one aligned field pair at a time* and answers a bounded comparison question. Structure is already resolved, so the judge never has to parse, resolve identifiers, or track time.

Because every score traces back to a specific unit, failures become diagnosable: an undefined identifier, a weak temporal overlap, a sound attached to the wrong shot, or a hallucinated attribute are each counted separately rather than averaged into one number.

| | Evaluated unit | Scoring operator | Coverage | Localization | Judge-stable |
|---|---|---|---|---|---|
| Whole-caption | Caption pair | Global LLM | ✅ | ❌ | ❌ |
| Probe-based | Sparse QA / cloze probes | LLM scorer | ❌ | ✅ | ❌ |
| Parse-then-score | Post-hoc parsed script | Parser + LLM | ⚠️ | ✅ | ⚠️ |
| **OmniCapBench** | **Native atomic units** | **Rule + bounded local LLM** | ✅ | ✅ | ✅ |

## Benchmark Statistics

786 densely annotated videos spanning 12.8 hours, 20 level-1 categories and 125 level-2 subcategories.

| Statistic | Avg. per video | Total |
|---|---|---|
| Videos (<1 min) | 61.32% | 482 |
| Videos (1–3 min) | 29.01% | 228 |
| Videos (3–5 min) | 9.67% | 76 |
| References | 7.40 | 5,818 |
| Shots | 14.53 | 11,419 |
| Subshots | 49.82 | 39,160 |
| Events | 8.32 | 6,537 |
| — of which dialogue | 6.83 | 5,370 |

## Evaluation Framework

A video `V` is formalized as a structured state `S*(V) = (R, E, H)` built across three interdependent tracks.

| Track | What it encodes | Verifiable properties |
|---|---|---|
| **References** (`R`) | Persistent identities for people, objects, and scenes, each with a unique identifier and an appearance anchor | Identity permanence across cuts, appearance fidelity |
| **Events** (`E`) | Auditory events and spoken dialogue with continuous time ranges | Sound recovery, temporal boundaries, transcription |
| **Shots** (`H`) | The visual timeline split into camera shots and finer **subshots**, each linking back to references and events | Shot recovery, boundary precision, audio–visual binding |

A model is evaluated through three tasks:

1. **Native unit generation** — the model emits the three tracks directly as a structured JSON object, not a paragraph.
2. **Structural verification and matching** — deterministic checks discard units with undefined identifiers, reversed time spans, or impossible event–shot links; the survivors are aligned to the ground truth.
3. **Localized semantic scoring** — the LLM judge compares only aligned content fields.

For open-vocabulary fields (references and subshots) scoring is **bidirectional**: the recall direction takes the ground truth as the subject and penalizes omissions, while the precision direction takes the prediction as the subject and penalizes hallucinations. This stops a model from gaming the score with either terse answers or speculative detail dumps.

## Metrics

Rule-based structural metrics and localized semantic metrics form two independent tables; no aggregate total score is reported, so every number stays attributable to one component of the video state.

**Rule-based structural metrics.** `SGC` is an independent schema prerequisite rather than a quality score.

| Category | Metric | What it measures |
|---|---|---|
| Gate | `SGC` | Schema gate compliance — is the output complete enough to be scored at all |
| Visual | `RefUse` | Whether references are correctly used inside shots, events, and subshots |
| Visual | `CCC` | Cross-shot coreference consistency — identity permanence across cuts |
| Visual | `Ref Subject F1` / `Ref Scene F1` | Detection of subject (person, animal, object) and scene references |
| Visual | `Shot F1` / `Shot tIoU` | Shot recovery and boundary precision |
| Visual | `Subshot F1` / `Subshot tIoU` | Fine-grained visual state recovery and boundary precision |
| Audio | `Event F1` / `Event tIoU` | Sound event recovery and temporal alignment |
| Audio-Visual | `Speaker F1` | Whether dialogue is attributed to the correct visual speaker |
| Audio-Visual | `EVSA F1` | Event–shot association — are sounds bound to concurrent visual shots |

**Localized semantic metrics**, computed exclusively on structurally aligned units:

| Category | Metric | What it measures |
|---|---|---|
| Visual | `Shot` | Camera and transition fact fidelity |
| Visual | `Subject Recall` / `Subject Precision` | Subject appearance description, bidirectionally |
| Visual | `Scene Recall` / `Scene Precision` | Scene description, bidirectionally |
| Visual | `Subshot Recall` / `Subshot Precision` | Fine-grained visual state description, bidirectionally |
| Audio | `Dialogue` | Rule-based dialogue line accuracy (normalized WER) |
| Audio | `Non-dialogue` | Semantic quality of non-dialogue event descriptions |

Exact formulations, matching thresholds, and output field names: **[docs/METRICS.md](docs/METRICS.md)**.

## Installation

Python 3.8+ and FFmpeg are required.

```bash
git clone https://github.com/OmniCapBench/OmniCapBench.git
cd OmniCapBench

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

For local GPU inference of open-source checkpoints:

```bash
pip install -e ".[local]"
pip install vllm            # optional, for serving large omnimodal models
```

FFmpeg is used for all video and audio preprocessing:

```bash
sudo apt-get install ffmpeg      # Debian / Ubuntu
brew install ffmpeg              # macOS
ffmpeg -version                  # verify
```

## Download the Benchmark

```bash
bash scripts/download_dataset.sh data/OmniCapBench
```

This pulls roughly 12 GB from [🤗 OmniCapBench/OmniCapBench](https://huggingface.co/datasets/OmniCapBench/OmniCapBench) and lays it out as:

```text
data/OmniCapBench/
├── eval.jsonl              # 786 annotation records, field name: video_path
└── test/
    ├── metadata.jsonl      # same records, field name: file_name (Dataset Viewer)
    └── videos/             # 786 media files
```

Video paths in the annotations are relative, so point the toolkit at the media root with `--video-root data/OmniCapBench/test` (or `OMNICAP_EVAL_VIDEO_ROOT`). The `video_path` field is never rewritten, because it is the key that joins predictions back to the ground truth.

## Quick Start

Verify the installation offline, without an API key, a GPU, or the video files:

```bash
python examples/run_metrics_demo.py
```

This scores the bundled sample predictions against the bundled annotations and prints the full rule-based metric table. The sample predictions are copies of the ground truth, so this is an *oracle sanity check* — every metric should land at or near 1.0. Reference matching falls back to exact identifier matching here; LLM-assisted matching activates once a judge endpoint is configured.

## Evaluating a Model

### 1. Configure credentials

```bash
cp .env.example .env     # then fill in only the keys you need
```

You only need keys for the models and judge you actually use. See [.env.example](.env.example) for the full list.

### 2. Generate structured predictions

```bash
VIDEO_ROOT=data/OmniCapBench/test WORKERS=16 \
  bash scripts/run_inference.sh gemini_25 data/OmniCapBench/eval.jsonl output/eval
```

Writes to `output/eval/gemini_25/`:

```text
results.jsonl      successful predictions
errors.jsonl       failed samples with tracebacks
progress.json      checkpoint; rerunning resumes where it stopped
```

Add `--dry-run` to the underlying command to preview what would be processed:

```bash
python -m omnicapbench_eval.run_eval --model gemini_25 \
  --input data/OmniCapBench/eval.jsonl --dry-run
```

### 3. Score the predictions

```bash
JUDGE_BASE_URL=https://api.openai.com/v1 \
JUDGE_MODEL=gpt-4o \
JUDGE_API_KEY=$OPENAI_API_KEY \
  bash scripts/eval_all_metrics.sh all data/OmniCapBench/eval.jsonl output/eval/gemini_25
```

Modes are `all`, `rule-based`, `llm-only`, and `convert-only`. The judge endpoint also drives LLM-assisted reference matching inside the rule-based metrics, which keeps `Ref`, `RefUse`, and `CCC` consistent with the semantic scores.

Results land in the prediction directory:

```text
metrics_summary.json          rule-based structural metrics
metrics_summary_detail.jsonl  per-video breakdown
llm_judge_scores.json         localized semantic metrics
report_tables.md              both result tables, rendered
```

### 4. Compare models side by side

```bash
python -m omnicapbench_eval.report_tables \
  Gemini-2.5-Pro=output/eval/gemini_25 \
  Qwen3-Omni=output/eval/qwen3_omni_instruct \
  --output output/report
```

Any OpenAI-compatible endpoint works as the judge — OpenAI, OpenRouter, DashScope, or a self-hosted vLLM server. Because the judge only performs bounded local comparisons, scores stay stable across judge strengths; see [docs/COOKBOOK.md](docs/COOKBOOK.md) for a judge-sensitivity walkthrough.

## Results

Rule-based structural evaluation, macro-averaged across videos (%):

| Model | SGC | RefUse | CCC | Ref Subject F1 | Ref Scene F1 | Shot F1 | Shot tIoU | Subshot F1 | Subshot tIoU | Event F1 | Event tIoU | Speaker F1 | EVSA F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemini 3.1-Pro | **97.36** | 70.39 | 34.43 | 78.60 | 82.68 | **76.24** | 70.74 | 65.27 | 48.76 | **63.32** | **71.92** | 91.18 | **51.46** |
| Gemini 2.5-Pro | 96.73 | **70.75** | **37.81** | **80.70** | **83.65** | 70.93 | **72.58** | **66.69** | **49.18** | 58.52 | 69.35 | 89.72 | 43.74 |
| Qwen3.5-Omni-Plus | 96.09 | 66.92 | 29.55 | 73.57 | 79.97 | 71.25 | 72.35 | 65.69 | 48.29 | 53.92 | 67.77 | **91.20** | 40.20 |
| Qwen3.5-Omni-Flash | 95.58 | 63.10 | 24.31 | 72.27 | 66.07 | 65.65 | 68.99 | 58.94 | 46.52 | 50.68 | 69.34 | 89.91 | 35.58 |
| Qwen3-Omni-Instruct | 89.02 | 40.96 | 9.85 | 54.84 | 64.55 | 49.98 | 67.92 | 29.51 | 47.95 | 41.70 | 40.99 | 88.35 | 13.33 |
| Qwen3-Omni-Captioner | 90.77 | 44.96 | 11.51 | 54.36 | 57.08 | 52.29 | 70.80 | 39.51 | **49.69** | 41.31 | 52.60 | 88.68 | 17.42 |

Localized semantic evaluation on structurally aligned units (%):

| Model | Shot | Subject Recall | Subject Precision | Scene Recall | Scene Precision | Subshot Recall | Subshot Precision | Dialogue | Non-dialogue |
|---|---|---|---|---|---|---|---|---|---|
| Gemini 3.1-Pro | **59.05** | 43.78 | **67.07** | 51.78 | **75.62** | **53.29** | 70.92 | 91.18 | **41.49** |
| Gemini 2.5-Pro | 55.58 | **51.53** | 57.08 | **56.80** | 65.12 | 53.05 | 71.21 | 89.72 | 37.38 |
| Qwen3.5-Omni-Plus | 54.17 | 45.41 | 62.84 | 50.75 | 69.80 | 50.22 | 70.66 | **91.20** | 36.82 |
| Qwen3.5-Omni-Flash | 47.66 | 43.87 | 65.59 | 44.41 | 71.26 | 44.08 | 67.27 | 89.91 | 28.12 |
| Qwen3-Omni-Instruct | 34.63 | 40.31 | 62.02 | 36.07 | 70.73 | 29.42 | 57.84 | 88.35 | 20.51 |
| Qwen3-Omni-Captioner | 36.08 | 42.00 | 50.96 | 38.07 | 54.53 | 34.01 | 46.05 | 88.68 | 19.65 |

Three findings stand out. Schema adherence is essentially solved (`SGC` above 95% for frontier models), yet **identity permanence is not**: `CCC` falls to 37.81% for the strongest model and near 10% for open-source ones, a deficit invisible to metrics that only check whether an entity was mentioned once. Models transcribe speech well (`Dialogue` above 88% everywhere) but **fail to bind sounds to their visual sources**, with Gemini 3.1-Pro dropping from 63.32% `Event F1` to 51.46% `EVSA F1`. Finally, restricting semantics to aligned units exposes a **precision–recall trade-off** that fluent prose normally hides: Gemini 3.1-Pro omits detail to stay safe (67.07% subject precision against 43.78% recall), while weaker models inflate recall by hallucinating attributes.

## Supported Models

| Model | Access | Configuration |
|---|---|---|
| Gemini 3.1-Pro / 2.5-Pro | Cloud | `GEMINI_API_KEY` |
| Qwen3.5-Omni-Plus / Flash | Cloud | `OMNICAP_EVAL_QWEN35_API_KEY` + base URL |
| GPT / Claude / 200+ others | Cloud | `OPENROUTER_API_KEY` + model name |
| Qwen3-Omni-30B-A3B (Instruct / Thinking / Captioner) | Local | vLLM — `bash scripts/serve_vllm.sh` |
| Qwen2.5-Omni-7B | Local | vLLM |
| ASID-Captioner, AVoCaDO, TimeChat-Captioner, UGC-VideoCaptioner | Local | Transformers + `HF_TOKEN` |

Download open-source checkpoints with `python -m omnicapbench_eval.download_models --all`.

Captioners fine-tuned toward free-form prose follow structured-output instructions poorly. For those, convert their output into MTSS units first via `CONVERT_MODEL_KIND=<name> bash scripts/eval_all_metrics.sh convert-only ...`, and read the resulting scores as a parse-then-score baseline rather than native structured generation.

## Caption Generation Pipeline

The same repository contains the pipeline used to build the benchmark annotations, so the structured tracks can be regenerated or extended to new videos. It runs an iterative local-refinement loop — **generate → evaluate & validate → local refine → select** — in the dependency order References → Events → Shots. A programmatic validator enforces schema constraints, and refinement only enriches text while identifiers, time boundaries, and cross-links stay invariant.

```bash
python -m omnicapbench_pipeline.run \
  --input examples/generation_input_sample.json \
  --output output/generation \
  --workers 4
```

Details and model wiring: [docs/COOKBOOK.md](docs/COOKBOOK.md).

## Project Structure

```text
OmniCapBench/
├── omnicapbench_eval/            # Evaluation toolkit
│   ├── run_eval.py               # Structured inference over the benchmark
│   ├── inference_engine.py       # Per-family video sampling and prompting
│   ├── mtss_metrics.py           # Rule-based structural metrics
│   ├── mtss_llm_judge.py         # Localized semantic scoring
│   ├── report_tables.py          # Renders both result tables
│   ├── mtss_transformer_convert.py  # Free-form caption → MTSS units
│   ├── native_vs_posthoc.py      # Native generation vs. post-hoc parsing audit
│   ├── download_models.py        # Open-source checkpoint downloader
│   └── mtss_generate_prompts/    # Structured generation prompts
│
├── omnicapbench_pipeline/        # Annotation construction pipeline
│   ├── run.py                    # CLI entry point
│   ├── pipeline.py               # Local-refinement loop
│   ├── stages/                   # Generate / evaluate / refine / select
│   ├── validators.py             # Schema and cross-link enforcement
│   └── prompts/                  # Per-stage prompt builders
│
├── scripts/                      # Shell entry points
├── examples/                     # Offline demo, sample annotations and predictions
├── docs/                         # Metrics reference and cookbook
└── tests/                        # Offline test suite
```

Run the offline tests with `pytest`; they touch only local files and make no API calls.

## Citation

```bibtex
@misc{omnicapbench2026,
  title        = {OmniCapBench: A Deep-Structured Evaluation Framework for Fine-Grained Audio-Visual Captioning},
  author       = {OmniCapBench Team},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/OmniCapBench/OmniCapBench}}
}
```

## License

The evaluation code in this repository is released under the **MIT License**; see [LICENSE](LICENSE).

The benchmark videos are distributed separately and are aggregated from multiple upstream collections, each with its own license and redistribution terms. Review the terms of the source collections before redistribution or commercial use, and treat the data as real-world video that may contain faces, speech, and other identifying content.
