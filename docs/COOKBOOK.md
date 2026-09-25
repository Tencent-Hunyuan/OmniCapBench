# OmniCapBench Cookbook

Runnable recipes for the evaluation toolkit and the annotation pipeline. Metric formulations live in [METRICS.md](METRICS.md).

## Contents

* [1. Data Format](#1-data-format)
* [2. Evaluating a Model](#2-evaluating-a-model)
* [3. Judge Configuration](#3-judge-configuration)
* [4. Evaluating Free-Form Captioners](#4-evaluating-free-form-captioners)
* [5. Local Model Serving](#5-local-model-serving)
* [6. Auditing the Evaluation Paradigm](#6-auditing-the-evaluation-paradigm)
* [7. Regenerating Annotations](#7-regenerating-annotations)
* [8. Troubleshooting](#8-troubleshooting)

## 1. Data Format

### Annotation records

Each line of `eval.jsonl` is one video. The evaluation reads `video_path` as the record identity and `mtss_caption` as the ground-truth state.

| Field | Description |
|---|---|
| `video_path` | Video location, relative to the media root |
| `duration_sec`, `duration_bucket` | Duration and coarse bucket (`<1min`, `1-3min`, `3-5min`) |
| `num_shots`, `num_events`, `num_main_characters`, `num_main_scenes` | Annotation counts |
| `category_l1`, `category_l2` | Content categories |
| `shot_time` | Shot boundary list, `[[start, end], ...]` |
| `mtss_caption` | Structured state: `references`, `events`, `shots` |

The `mtss_caption` object holds the three tracks:

```json
{
  "references": [
    {
      "ref_id": "PERSON_1",
      "type": "person",
      "semantic_description": "The news anchor presenting from the studio.",
      "appearance_anchor": {
        "id_features": {
          "detail_description": "A woman with short golden hair reaching her chin, wearing a dark blazer."
        }
      }
    }
  ],
  "events": [
    {
      "event_id": "EVENT_1",
      "type": "dialogue",
      "time_range": [10.5, 12.0],
      "content": {"line": "Good evening.", "speaker": "PERSON_1", "description": "..."}
    }
  ],
  "shots": [
    {
      "shot_id": "SHOT_1",
      "time_range": [0.0, 6.56],
      "shot_transfer": "hard cut",
      "camera": "medium shot, eye-level, static",
      "visual_description": "...",
      "active_events": ["EVENT_1"],
      "references_in_shot": ["PERSON_1"],
      "sub_shots": [
        {"sub_shot_time": [0.0, 2.1], "sub_shot_content": "[PERSON_1] looks toward the camera."}
      ]
    }
  ]
}
```

### Prediction records

Predictions are JSONL with one record per video. Only `status == "success"` records are scored, and they are joined to the ground truth on an exact `video_path` match.

```json
{
  "video_path": "videos/ytb_2tX_148en_Y.mp4",
  "status": "success",
  "model_prediction": {"references": [], "events": [], "shots": []}
}
```

`run_eval.py` writes this shape automatically. If you evaluate a model that is not wired into the toolkit, emit the same shape yourself and skip straight to `scripts/eval_all_metrics.sh`.

### Relative paths

`video_path` is never rewritten, because it is the join key. To read media from a downloaded dataset, pass the media root separately:

```bash
python -m omnicapbench_eval.run_eval --model gemini_25 \
  --input data/OmniCapBench/eval.jsonl \
  --video-root data/OmniCapBench/test
```

The resolved location is stored in `media_path` for the inference engine only.

## 2. Evaluating a Model

### End to end

```bash
bash scripts/download_dataset.sh data/OmniCapBench

VIDEO_ROOT=data/OmniCapBench/test WORKERS=16 \
  bash scripts/run_inference.sh gemini_25 data/OmniCapBench/eval.jsonl output/eval

JUDGE_BASE_URL=https://api.openai.com/v1 JUDGE_MODEL=gpt-4o JUDGE_API_KEY=$OPENAI_API_KEY \
  bash scripts/eval_all_metrics.sh all data/OmniCapBench/eval.jsonl output/eval/gemini_25
```

### Partial and resumable runs

Inference checkpoints after every sample, so rerunning the same command resumes.

```bash
# Preview without calling any model
python -m omnicapbench_eval.run_eval --model gemini_25 --input data/OmniCapBench/eval.jsonl --dry-run

# Shard across machines
python -m omnicapbench_eval.run_eval --model gemini_25 --start 0 --end 200
python -m omnicapbench_eval.run_eval --model gemini_25 --start 200 --end 400

# Start over
python -m omnicapbench_eval.run_eval --model gemini_25 --no-resume
```

### Rule-based metrics only

Structural metrics run without any judge endpoint. Reference matching then falls back to exact identifier matching, which only credits predictions that reuse the ground-truth identifiers verbatim.

```bash
bash scripts/eval_all_metrics.sh rule-based data/OmniCapBench/eval.jsonl output/eval/gemini_25
```

### Reporting

```bash
python -m omnicapbench_eval.report_tables \
  Gemini-3.1-Pro=output/eval/gemini_31 \
  Gemini-2.5-Pro=output/eval/gemini_25 \
  Qwen3-Omni=output/eval/qwen3_omni_instruct \
  --output output/report
```

Writes `report_tables.md` and `report_tables.json` with both result tables side by side.

## 3. Judge Configuration

Any OpenAI-compatible endpoint works. The judge only performs bounded local comparisons, so it does not need frontier-level reasoning.

```bash
# OpenAI
JUDGE_BASE_URL=https://api.openai.com/v1 JUDGE_MODEL=gpt-4o JUDGE_API_KEY=sk-...

# OpenRouter
JUDGE_BASE_URL=https://openrouter.ai/api/v1 JUDGE_MODEL=google/gemini-2.5-pro JUDGE_API_KEY=sk-or-...

# DashScope
JUDGE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 JUDGE_MODEL=qwen-plus JUDGE_API_KEY=sk-...

# Self-hosted vLLM
JUDGE_BASE_URL=http://localhost:8000/v1 JUDGE_MODEL=Qwen3-32B JUDGE_API_KEY=EMPTY
```

The same endpoint drives reference matching in the rule-based metrics, so `Ref`, `RefUse`, and `CCC` stay consistent with the semantic table. To verify judge stability, score the same prediction file with a weak and a strong judge into different output directories and compare:

```bash
for judge in Qwen3-32B gpt-4o; do
  JUDGE_MODEL=$judge bash scripts/eval_all_metrics.sh llm-only \
    data/OmniCapBench/eval.jsonl output/eval/gemini_25
  mv output/eval/gemini_25/llm_judge_scores.json output/eval/gemini_25/judge_$judge.json
done
```

Name result files after the judge model so different configurations never overwrite each other.

## 4. Evaluating Free-Form Captioners

Captioners fine-tuned toward prose follow structured-output instructions poorly. Convert their output into MTSS units first, then score the conversion:

```bash
CONVERT_MODEL_KIND=avocado \
JUDGE_BASE_URL=https://api.openai.com/v1 JUDGE_MODEL=gpt-4o JUDGE_API_KEY=$OPENAI_API_KEY \
  bash scripts/eval_all_metrics.sh all data/OmniCapBench/eval.jsonl output/eval/avocado
```

The converter writes `results_mtss_converted.jsonl`, which the metrics then pick up in place of `results.jsonl`. Supported source kinds are `asid_caption_3b`, `asid_caption_7b`, `avocado`, `timechat_captioner`, and `ugc_videocaptioner`.

Read these numbers as a **parse-then-score baseline**, not as native structured generation: the conversion step adds its own extraction errors, which is precisely the fragility the benchmark is designed to avoid.

## 5. Local Model Serving

Download checkpoints, then serve them through vLLM's OpenAI-compatible API:

```bash
export HF_TOKEN=your-token
python -m omnicapbench_eval.download_models --all

bash scripts/serve_vllm.sh omnicapbench_eval/models/Qwen3-Omni-30B-A3B-Instruct 8903
```

In another shell:

```bash
OMNICAP_EVAL_QWEN3_BASE_URL=http://localhost:8903/v1 \
VIDEO_ROOT=data/OmniCapBench/test WORKERS=4 \
  bash scripts/run_inference.sh qwen3_omni_instruct data/OmniCapBench/eval.jsonl output/eval
```

Match `WORKERS` to the server's `MAX_NUM_SEQS` (4 for the 30B model, 8 for the 7B model). Models in the `qwen25_omni` family are limited to videos under 60 seconds and longer videos are skipped automatically.

Video sampling differs per model family and is defined in `omnicapbench_eval/eval_config.py`:

| Family | Sampling | Long videos |
|---|---|---|
| `closed_source` | fps 2, no frame cap | Single pass |
| `qwen3_omni` | fps 2, max 192 frames | Two passes above 60s: reference bank, then per-shot-group events and shots |
| `qwen25_omni` | fps 2, max 192 frames | Skipped above 60s |

## 6. Auditing the Evaluation Paradigm

`native_vs_posthoc.py` reproduces the comparison between native structured generation and post-hoc parsing of a dense caption:

```bash
python -m omnicapbench_eval.native_vs_posthoc --stage caption --model qwen3_omni_instruct
python -m omnicapbench_eval.native_vs_posthoc --stage parse   --model qwen3_omni_instruct
python -m omnicapbench_eval.native_vs_posthoc --stage compare --model qwen3_omni_instruct
```

Stage `caption` reuses the inference infrastructure to produce dense prose, `parse` converts it into MTSS units with an LLM parser, and `compare` scores both pipelines against the same ground truth.

## 7. Regenerating Annotations

The annotation pipeline builds the three tracks in dependency order (References → Events → Shots) through a **generate → evaluate & validate → local refine → select** loop. A programmatic validator enforces schema constraints, and refinement only enriches wording while identifiers, time boundaries, and cross-links stay invariant.

```bash
# Single video, useful for inspecting stage outputs
python -m omnicapbench_pipeline.run \
  --input examples/generation_input_sample.json \
  --output output/generation \
  --single 0

# Batch
python -m omnicapbench_pipeline.run \
  --input examples/generation_input_sample.json \
  --output output/generation \
  --workers 4
```

Input records need only a video path and shot boundaries:

```json
[{"DID": "example_001", "video_path": "videos/example.mp4", "audio_path": "", "shot_time": [[0.0, 5.0], [5.0, 10.0]]}]
```

Useful options: `--limit N` for the first N entries, `--sample N` for stratified sampling across duration buckets, `--no-resume` to ignore checkpoints, and `--model` to override the model for every stage.

Configure the generation models through the environment:

```bash
# Local Qwen-Omni via an OpenAI-compatible server (default)
export OMNICAP_QWEN_OMNI_BASE_URL=http://localhost:8000/v1
export OMNICAP_QWEN_OMNI_API_KEY=EMPTY

# Or Gemini
export GEMINI_API_KEY=your-key

# Or OpenRouter
export OPENROUTER_API_KEY=your-key
```

Long videos are processed in shot chunks; `OMNICAP_SHOT_CHUNK_SIZE` and `OMNICAP_SHOT_CHUNK_THRESHOLD` control the split.

## 8. Troubleshooting

**`ffmpeg not found`** — install FFmpeg, or point `OMNICAP_EVAL_FFMPEG_PATH` and `OMNICAP_EVAL_FFPROBE_PATH` at your binaries.

**All metrics are 0 and the log says 0 successful predictions** — the ground truth and predictions are joined on an exact `video_path` match. Confirm the prediction file carries the same `video_path` strings as the annotation file, and that records have `status: "success"`.

**`Ref`, `RefUse`, and `CCC` look implausibly low** — LLM-assisted reference matching is inactive, so only verbatim identifier reuse counts. Set `JUDGE_BASE_URL` and `JUDGE_API_KEY`, or run through `scripts/eval_all_metrics.sh`, which forwards them as `REF_MATCH_*`.

**Videos are not found** — annotation paths are relative to the dataset root. Pass `--video-root data/OmniCapBench/test` or set `OMNICAP_EVAL_VIDEO_ROOT`.

**Inference is rate-limited** — lower `WORKERS`. Remote API presets already retry with backoff; local vLLM presets retry less because their failures are usually deterministic.

**A run died partway through** — rerun the identical command. Checkpoints live in `progress.json` and `results.jsonl`, and only successful samples are skipped, so failures are retried.
