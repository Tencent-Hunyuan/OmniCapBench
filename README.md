# OmniCapBench: A Deep-Structured Evaluation Framework for Fine-Grained Audio-Visual Captioning

[📖 Metrics Reference](docs/METRICS.md) · [中文说明](README_zh.md)

## News

* **[2026/09]** 🎉 **OmniCapBench** is now open-source: the annotation pipeline for building a deep-structured evaluation set, plus the full scoring toolkit.

## Contents

* [Introduction](#introduction)
* [Evaluation Units](#evaluation-units)
* [Metrics](#metrics)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Step 1 — Build an Evaluation Set](#step-1--build-an-evaluation-set)
* [Step 2 — Score a Model](#step-2--score-a-model)
* [Supported Models](#supported-models)
* [Citation](#citation)
* [License](#license)

## Introduction

**OmniCapBench** reframes audio–visual caption evaluation as a *deep-structured diagnostic* task. Instead of scoring a free-form paragraph with a global LLM judge, it shifts the prediction target to sets of **atomic, verifiable evaluation units** across three native tracks: entity references, visual shots, and audio events.

This resolves a trade-off existing caption benchmarks cannot escape. Whole-caption protocols cover the whole video but compress everything into an opaque scalar that hides local errors. Probe-based protocols localize errors but cover only sparse points. Both then hand dense, monolithic text to an unconstrained LLM judge, which makes scores swing with judge strength.

OmniCapBench keeps full coverage *and* exact localization by splitting the scoring operator in two:

1. **Deterministic verification.** Rules check schema validity, identifier scoping, time bounds, and cross-track links, then align predicted units to the ground truth by temporal IoU (shots, events) or a bounded LLM matcher (references, subshots).
2. **Localized semantic checking.** An LLM judge sees only *one aligned field pair at a time*. Structure is already resolved, so it never has to parse text, resolve identifiers, or track time.

Because every score traces back to a specific unit, failures become diagnosable: an undefined identifier, a weak temporal overlap, a sound attached to the wrong shot, or a hallucinated attribute are each counted separately rather than averaged away.

The paper applies this framework to 786 densely annotated videos (12.8 hours, 20 level-1 and 125 level-2 categories, yielding 5,818 references, 11,419 shots, 39,160 subshots, and 6,537 events). This repository contains the pipeline that produced those annotations and the metrics that scored them, so the same protocol can be reproduced on any video collection.

## Evaluation Units

A video `V` is formalized as a structured state `S*(V) = (R, E, H)` across three interdependent tracks. This schema is what the pipeline emits and what the metrics consume.

| Track                        | What it encodes                                                                                                     | Verifiable properties                                    |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **References** (`R`) | Persistent identities for people, objects, and scenes, each with a unique identifier and an appearance anchor       | Identity permanence across cuts, appearance fidelity     |
| **Events** (`E`)     | Auditory events and spoken dialogue with continuous time ranges                                                     | Sound recovery, temporal boundaries, transcription       |
| **Shots** (`H`)      | The visual timeline split into camera shots and finer**subshots**, each linking back to references and events | Shot recovery, boundary precision, audio–visual binding |

For open-vocabulary fields (references and subshots) scoring is **bidirectional**: the recall direction takes the ground truth as the subject and penalizes omissions, while the precision direction takes the prediction as the subject and penalizes hallucinations. This stops a model from gaming the score with either terse answers or speculative detail dumps.

Field-level schema and the exact JSONL record shapes: **[docs/METRICS.md](docs/METRICS.md)**.

## Metrics

Rule-based structural metrics and localized semantic metrics form two independent tables; no aggregate total score is reported, so every number stays attributable to one component of the video state.

| Family       | Metrics                                                                         | What it measures                                                                        |
| ------------ | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Gate         | `SGC`                                                                         | Schema gate compliance — is the output complete enough to be scored at all             |
| Visual       | `RefUse`, `CCC`                                                             | Correct reference use inside shots/events/subshots, and identity permanence across cuts |
| Visual       | `Ref Subject F1`, `Ref Scene F1`                                            | Detection of subject (person, animal, object) and scene references                      |
| Visual       | `Shot F1`/`tIoU`, `Subshot F1`/`tIoU`                                   | Visual state recovery and temporal boundary precision                                   |
| Audio        | `Event F1`, `Event tIoU`                                                    | Sound event recovery and temporal alignment                                             |
| Audio-Visual | `Speaker F1`, `EVSA F1`                                                     | Dialogue speaker attribution, and whether sounds bind to concurrent shots               |
| Semantic     | `Shot`, `Subject`/`Scene`/`Subshot` R&P, `Dialogue`, `Non-dialogue` | Descriptive fidelity, computed only on structurally aligned units                       |

Exact formulations, matching thresholds, and output field names: **[docs/METRICS.md](docs/METRICS.md)**.

## Installation

Python 3.8+ and FFmpeg are required.

```bash
git clone <repository-url>
cd OmniCapBench

python -m venv .venv && source .venv/bin/activate
pip install -e .            # add ".[local]" for local GPU inference
```

## Quick Start

Verify the installation offline — no API key, GPU, or video file needed:

```bash
python examples/run_metrics_demo.py
```

This scores the bundled sample predictions against the bundled annotations, which also serve as a format reference for the two JSONL shapes the toolkit uses. The samples are copies of the ground truth, so it is an *oracle sanity check*: every metric should land at or near 1.0.

## Step 1 — Build an Evaluation Set

The annotation pipeline turns raw videos into the structured state described above. It runs an iterative local-refinement loop — **generate → evaluate & validate → local refine → select** — in the dependency order References → Events → Shots. A programmatic validator enforces schema constraints, and refinement only enriches text while identifiers, time boundaries, and cross-links stay invariant, so linguistic polish cannot introduce new hallucinations or corrupt a verified topology.

Input is one record per video, needing only a path and shot boundaries:

```json
[{"DID": "vid_001", "video_path": "data/videos/example.mp4", "audio_path": "", "shot_time": [[0.0, 5.0], [5.0, 10.0]]}]
```

```bash
# Single video, useful for inspecting each stage's output
python -m omnicapbench_pipeline.run \
  --input examples/generation_input_sample.json --output output/generation --single 0

# Batch
python -m omnicapbench_pipeline.run \
  --input my_videos.json --output output/generation --workers 4
```

Useful options: `--limit N` for the first N entries, `--sample N` for stratified sampling across duration buckets, `--no-resume` to ignore checkpoints, and `--model` to override the model for every stage. Configure the generation models through `OMNICAP_QWEN_OMNI_BASE_URL`, `GEMINI_API_KEY`, or `OPENROUTER_API_KEY`; see [.env.example](.env.example).

The result is an annotation set in the same JSONL shape as `examples/eval_sample.jsonl`, ready to be used as ground truth in step 2.

## Step 2 — Score a Model

```bash
cp .env.example .env     # fill in only the keys you need

# Generate structured predictions
VIDEO_ROOT=data/videos WORKERS=16 \
  bash scripts/run_inference.sh gemini_25 my_benchmark/eval.jsonl output/eval

# Score them
JUDGE_BASE_URL=https://api.openai.com/v1 JUDGE_MODEL=gpt-4o JUDGE_API_KEY=$OPENAI_API_KEY \
  bash scripts/eval_all_metrics.sh all my_benchmark/eval.jsonl output/eval/gemini_25

# Compare models side by side
python -m omnicapbench_eval.report_tables \
  Gemini-2.5-Pro=output/eval/gemini_25 \
  Qwen3-Omni=output/eval/qwen3_omni_instruct --output output/report
```

Annotation `video_path` entries may be relative; `--video-root` resolves them against a media directory. The `video_path` field itself is never rewritten, because it is the key that joins predictions back to the ground truth.

Inference checkpoints after every sample, so rerunning the same command resumes. Scoring modes are `all`, `rule-based`, `llm-only`, and `convert-only`; results land in the prediction directory as `metrics_summary.json`, `metrics_summary_detail.jsonl`, `llm_judge_scores.json`, and `report_tables.md`.

Any OpenAI-compatible endpoint works as the judge — OpenAI, OpenRouter, DashScope, or a self-hosted vLLM server. The same endpoint also drives LLM-assisted reference matching inside the rule-based metrics, which keeps `Ref`, `RefUse`, and `CCC` consistent with the semantic scores. Because the judge only performs bounded local comparisons, scores stay stable across judge strengths. Without a judge endpoint the rule-based metrics still run, but reference matching falls back to exact identifier matching.

Every script accepts `--help`, and `run_eval.py` additionally supports `--dry-run` to preview, `--start`/`--end` to shard across machines, and `--no-resume` to start fresh.

## Supported Models

| Model                                                           | Access | Configuration                              |
| --------------------------------------------------------------- | ------ | ------------------------------------------ |
| Gemini 3.1-Pro / 2.5-Pro                                        | Cloud  | `GEMINI_API_KEY`                         |
| Qwen3.5-Omni-Plus / Flash                                       | Cloud  | `OMNICAP_EVAL_QWEN35_API_KEY` + base URL |
| GPT / Claude / 200+ others                                      | Cloud  | `OPENROUTER_API_KEY` + model name        |
| Qwen3-Omni-30B-A3B (Instruct / Thinking / Captioner)            | Local  | vLLM —`bash scripts/serve_vllm.sh`      |
| Qwen2.5-Omni-7B                                                 | Local  | vLLM                                       |
| ASID-Captioner, AVoCaDO, TimeChat-Captioner, UGC-VideoCaptioner | Local  | Transformers +`HF_TOKEN`                 |

Download open-source checkpoints with `python -m omnicapbench_eval.download_models --all`.

Captioners fine-tuned toward free-form prose follow structured-output instructions poorly. For those, convert their output into MTSS units first with `CONVERT_MODEL_KIND=<name> bash scripts/eval_all_metrics.sh convert-only ...`, and read the resulting scores as a parse-then-score baseline rather than native structured generation.

Run the offline tests with `pytest`; they touch only local files and make no API calls.

## Citation

```bibtex
@misc{omnicapbench2026,
  title  = {OmniCapBench: A Deep-Structured Evaluation Framework for Fine-Grained Audio-Visual Captioning},
  author = {OmniCapBench Team},
  year   = {2026}
}
```

## License

The code in this repository is released under the **Apache License 2.0**; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

No video data or annotations are distributed here. If you build an evaluation set from third-party video collections, each source carries its own license and redistribution terms — review them before redistribution or commercial use, and treat the material as real-world video that may contain faces, speech, and other identifying content.
