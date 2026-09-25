#!/usr/bin/env python3
"""
MTSS Eval - Experiment 3: State-native vs Post-hoc Validator

Proves that MTSS native state prediction cannot be replaced by
"structured caption + LLM parser" (post-hoc conversion).

Pipeline A (Native): video -> model -> MTSS state  (already done in run_eval.py)
Pipeline B (Post-hoc): video -> model -> dense caption -> LLM parser -> MTSS state

This script implements Pipeline B and compares with Pipeline A results.

Usage:
    # Step 1: Generate dense captions (reuses existing inference infra)
    python native_vs_posthoc.py --stage caption --model qwen3_omni_instruct [--workers 1]

    # Step 2: Parse captions into MTSS states via LLM
    python native_vs_posthoc.py --stage parse --model qwen3_omni_instruct

    # Step 3: Evaluate and compare native vs post-hoc
    python native_vs_posthoc.py --stage compare --model qwen3_omni_instruct

    # Or run all stages:
    python native_vs_posthoc.py --stage all --model qwen3_omni_instruct
"""

import argparse
import json
import logging
import os
import sys
import time
import fcntl
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("native_vs_posthoc")

# ======================================================================
# Dense caption prompt (for Pipeline B Step 1)
# ======================================================================

DENSE_CAPTION_SYSTEM_PROMPT = """You are an expert video captioner. Watch the video carefully and produce a comprehensive, detailed dense caption that describes EVERYTHING you observe.

Your caption must cover ALL of the following aspects in a single continuous text:

1. **Scene & Setting**: Describe the overall scene, location, environment, lighting, and visual style.

2. **Characters/Entities**: For EVERY person, animal, or significant object:
   - Physical appearance (age, gender, clothing, hair, distinguishing features)
   - Their role in the video
   - When they first appear and when they leave

3. **Actions & Events**: Describe ALL actions chronologically:
   - What happens, who does it, to whom
   - Include timestamps: "At 0:05, the man picks up the cup"
   - Cover dialogue/speech content if any
   - Cover sound effects and music

4. **Temporal Structure**: Describe shot-by-shot:
   - When camera cuts happen
   - Camera angles and movements
   - Transitions between scenes

5. **Audio**: Describe all audio:
   - Dialogue (with speaker attribution and exact words if possible)
   - Sound effects
   - Background music
   - Ambient sounds

Be as detailed and specific as possible. Use timestamps throughout.
Do NOT use any structured format — write everything as natural flowing text paragraphs.
Output ONLY the caption text, no headers or formatting."""


# ======================================================================
# LLM Parser prompt (for Pipeline B Step 2)
# ======================================================================

PARSER_SYSTEM_PROMPT = """You are a structured data extraction expert. You will receive a dense video caption text and a list of pre-defined shot time boundaries. Your task is to convert this caption into a structured MTSS (MultiTrack SceneScript) JSON object.

The output JSON must follow this exact schema:
{
  "scene_description": "string — one-sentence summary of the entire video",
  "global_style": "string — visual style, quality, color tone",
  "global_audio": {"background_music": "string|null", "ambient_sound": "string|null"},
  "references": [
    {
      "ref_id": "TYPE_N (e.g., PERSON_1, OBJECT_1, SCENE_1)",
      "type": "person|animal|object|scene",
      "semantic_description": "who/what this entity is",
      "appearance_anchor": {
        "id_features": {
          "detail_description": "detailed visual appearance",
          "timestamp": "best-view timestamp in seconds, or empty for scenes"
        },
        "attributes": {"clothing": "...", "hairstyle": "...", "accessories": "..."} // only for person, else {}
      }
    }
  ],
  "events": [
    {
      "event_id": "TYPE_N (e.g., DIALOGUE_1, SFX_1)",
      "type": "dialogue|sfx|music",
      "time_range": [start_sec, end_sec],
      "content": {
        "speaker": "ref_id of speaker (for dialogue/music)",
        "line": "exact spoken/sung text",
        "description": "detailed description of the audio event"
      }
    }
  ],
  "shots": [
    {
      "shot_id": "shot_N",
      "time_range": [start_sec, end_sec],
      "references_in_shot": ["PERSON_1", "OBJECT_2"],
      "active_events": ["DIALOGUE_1"],
      "visual_description": [
        {
          "sub_shot_time": [start_sec, end_sec],
          "sub_shot_content": "detailed visual description of this sub-shot"
        }
      ],
      "camera": {"shot_type": "...", "angle": "...", "movement": "..."},
      "shot_transfer": "description of transition to next shot"
    }
  ]
}

CRITICAL RULES:
1. Extract ALL entities mentioned in the caption as references
2. Extract ALL audio events (dialogue, sound effects, music) as events
3. Use the provided shot_time boundaries to create shots — do NOT invent new boundaries
4. Every ref_id used in shots/events MUST be defined in references
5. Every active_event in shots MUST be defined in events
6. Time ranges must be in seconds, precise to 0.1s
7. Output ONLY valid JSON, no explanation text

The pre-defined shot boundaries are:
{shot_time}

Dense caption to convert:
{caption}"""


# ======================================================================
# Stage 1: Generate dense captions
# ======================================================================

def run_dense_caption_inference(model_name: str, gt_path: str, output_dir: str,
                                 workers: int = 1, caption_base_url: str = None,
                                 indices_file: str = None, shard_id: int = None):
    """
    Run dense caption inference for Pipeline B.
    Reuses the same model but with a different prompt (dense caption instead of MTSS).

    Args:
        caption_base_url: Override the model's base_url (e.g. when vLLM is on a specific node).
        indices_file: Path to JSON file with list of indices (for multi-node sharding).
        shard_id: Shard ID for output file naming (avoids NFS concurrent write corruption).
    """
    from . import eval_config as config
    from . import video_utils
    from .run_eval import MODEL_PRESETS, load_input_data, load_completed_ids
    from .inference_engine import (
        _get_shared_client, build_content_for_closed_source,
        build_content_for_qwen_omni, build_content_for_transformers,
    )
    import tempfile

    model_cfg = MODEL_PRESETS.get(model_name)
    if not model_cfg:
        logger.error(f"Unknown model: {model_name}")
        return

    # Override base_url if provided (for connecting to a running vLLM server)
    if caption_base_url:
        model_cfg = dict(model_cfg)  # shallow copy
        model_cfg["base_url"] = caption_base_url
        logger.info(f"Using override base_url: {caption_base_url}")

    model_family = model_cfg.get("model_family", "closed_source")
    strategy = config.SAMPLING_STRATEGY.get(model_family, config.SAMPLING_STRATEGY["closed_source"])

    # Output paths
    caption_dir = os.path.join(output_dir, model_name, "posthoc")
    os.makedirs(caption_dir, exist_ok=True)
    caption_file = os.path.join(caption_dir, "dense_captions.jsonl")

    # Load data (with optional index-based shard selection)
    all_data = load_input_data(gt_path, indices_file=indices_file)
    if strategy.get("skip_long_video"):
        max_dur = strategy.get("max_duration", 60)
        all_data = [d for d in all_data if d.get("duration_sec", 0) <= max_dur]

    # When using shard_id, write to a separate shard file to avoid NFS concurrent write corruption
    if shard_id is not None:
        caption_file = os.path.join(caption_dir, f"dense_captions_shard_{shard_id}.jsonl")

    # Resume: skip completed (check both main file and all shard files)
    completed_paths = set()
    # Check main file
    main_caption_file = os.path.join(caption_dir, "dense_captions.jsonl")
    for check_file in [main_caption_file] + \
            [os.path.join(caption_dir, f) for f in os.listdir(caption_dir)
             if f.startswith("dense_captions_shard_") and f.endswith(".jsonl")]:
        if os.path.exists(check_file):
            with open(check_file, "r") as f:
                for line in f:
                    try:
                        d = json.loads(line.strip())
                        if d.get("status") == "success":
                            completed_paths.add(d.get("video_path", ""))
                    except:
                        pass
    remaining = [d for d in all_data if d.get("video_path", "") not in completed_paths]
    logger.info(f"Dense caption: {len(completed_paths)} done, {len(remaining)} remaining")

    if not remaining:
        logger.info("All dense captions already generated!")
        return

    # Get model client
    tag = f"{model_name}_dense_caption"
    client = _get_shared_client(model_cfg, tag)

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_cache", model_name)
    os.makedirs(cache_dir, exist_ok=True)

    for i, item in enumerate(remaining):
        video_path = item.get("video_path", "")
        video_url = item.get("video_url", "")
        vname = os.path.basename(video_path)
        logger.info(f"[{i+1}/{len(remaining)}] Dense caption for {vname}")

        try:
            # Build content (same as native, but with dense caption prompt)
            if model_family == "closed_source":
                content = build_content_for_closed_source(video_url, "")
            elif model_cfg.get("inference_backend") == "transformers":
                processed = video_utils.preprocess_video(
                    video_path, resolution=strategy["resolution"],
                    add_timestamp=True, cache_dir=cache_dir)
                if not processed or not os.path.exists(processed):
                    raise FileNotFoundError(f"Preprocessed video missing: {processed}")
                content = build_content_for_transformers(processed, "")
            else:
                processed = video_utils.preprocess_video(
                    video_path, resolution=strategy["resolution"],
                    add_timestamp=True, cache_dir=cache_dir)
                if not processed or not os.path.exists(processed):
                    raise FileNotFoundError(f"Preprocessed video missing: {processed}")
                content = build_content_for_qwen_omni(processed, "", cache_dir=cache_dir)

            # Inference with dense caption prompt
            response = client.inference(content, system_prompt=DENSE_CAPTION_SYSTEM_PROMPT)

            record = {
                "video_path": video_path,
                "video_url": video_url,
                "duration_sec": item.get("duration_sec", 0),
                "shot_time": item.get("shot_time", []),
                "status": "success" if response and len(response) > 50 else "failed",
                "dense_caption": response or "",
                "error": None if response else "empty_response",
            }
        except Exception as e:
            record = {
                "video_path": video_path,
                "status": "failed",
                "dense_caption": "",
                "error": str(e),
            }

        # Append atomically
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with open(caption_file, "a", encoding="utf-8") as f:
            f.write(line)

        if record["status"] == "success":
            sys.stderr.write(f"  ✅ {vname} ({len(record['dense_caption'])} chars)\n")
        else:
            sys.stderr.write(f"  ❌ {vname}: {record.get('error', '')}\n")

    logger.info(f"Dense captions saved to {caption_file}")


# ======================================================================
# Stage 2: Parse dense captions into MTSS via LLM
# ======================================================================

def run_parser(model_name: str, output_dir: str, parser_model: str = "Qwen3-VL-235B-Instruct",
               parser_base_url: str = "http://localhost:8000/v1",
               parser_workers: int = 8, parser_api_key: str = "EMPTY"):
    """
    Parse dense captions into MTSS states using an LLM parser.
    Uses requests (OpenAI-compatible API) to avoid openai SDK dependency.
    """
    import requests

    caption_dir = os.path.join(output_dir, model_name, "posthoc")
    caption_file = os.path.join(caption_dir, "dense_captions.jsonl")
    parsed_file = os.path.join(caption_dir, "posthoc_outputs.jsonl")
    error_file = os.path.join(caption_dir, "posthoc_parse_errors.jsonl")

    # Collect all caption files (main + shards)
    caption_files = []
    if os.path.exists(caption_file):
        caption_files.append(caption_file)
    for f in sorted(os.listdir(caption_dir)) if os.path.isdir(caption_dir) else []:
        if f.startswith("dense_captions_shard_") and f.endswith(".jsonl"):
            caption_files.append(os.path.join(caption_dir, f))

    if not caption_files:
        logger.error(f"No dense captions found at {caption_dir}")
        return

    # Load captions from all files (deduplicate by video_path)
    captions = []
    seen_paths = set()
    for cf in caption_files:
        with open(cf, "r") as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    vp = d.get("video_path", "")
                    if d.get("status") == "success" and d.get("dense_caption") and vp not in seen_paths:
                        captions.append(d)
                        seen_paths.add(vp)
                except:
                    pass
    logger.info(f"Loaded {len(captions)} successful dense captions from {len(caption_files)} files")

    # Resume: skip already parsed
    parsed_paths = set()
    if os.path.exists(parsed_file):
        with open(parsed_file, "r") as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    parsed_paths.add(d.get("video_path", ""))
                except:
                    pass
    remaining = [c for c in captions if c.get("video_path", "") not in parsed_paths]
    logger.info(f"Parser: {len(parsed_paths)} done, {len(remaining)} remaining")

    if not remaining:
        logger.info("All captions already parsed!")
        return

    # Clear old error file before retrying (errors will be re-written)
    if os.path.exists(error_file) and remaining:
        os.remove(error_file)

    api_url = f"{parser_base_url.rstrip('/')}/chat/completions"

    # Create a session that ignores proxy environment variables for direct API calls.
    session = requests.Session()
    session.trust_env = False  # Ignore http_proxy/https_proxy env vars
    if parser_api_key and parser_api_key != "EMPTY":
        session.headers.update({"Authorization": f"{parser_api_key}"})

    def parse_one(caption_item):
        vp = caption_item.get("video_path", "")
        shot_time = caption_item.get("shot_time", [])
        caption_text = caption_item.get("dense_caption", "")

        prompt = PARSER_SYSTEM_PROMPT.replace(
            "{shot_time}", json.dumps(shot_time, ensure_ascii=False)
        ).replace(
            "{caption}", caption_text
        )

        payload = {
            "model": parser_model,
            "messages": [
                {"role": "system", "content": "You are a structured data extraction expert."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 32768,
        }

        last_error = None
        for attempt in range(5):
            try:
                resp = session.post(api_url, json=payload, timeout=300)
                resp.raise_for_status()
                data = resp.json()
                raw_output = data["choices"][0]["message"]["content"]

                # Parse JSON from response
                parsed = _extract_json(raw_output)
                if parsed is None:
                    return {
                        "video_path": vp,
                        "parse_success": False,
                        "parse_error_type": "json_parse_failed",
                        "raw_parser_output": raw_output[:2000],
                        "output_state": {},
                    }

                return {
                    "video_path": vp,
                    "duration_sec": caption_item.get("duration_sec", 0),
                    "shot_time": shot_time,
                    "parse_success": True,
                    "parse_error_type": None,
                    "output_state": parsed,
                    "status": "success",
                }
            except Exception as e:
                last_error = e
                if attempt < 4:
                    sleep_time = min(60, 5 * (2 ** attempt))
                    time.sleep(sleep_time)

        return {
            "video_path": vp,
            "parse_success": False,
            "parse_error_type": "api_error",
            "error": str(last_error),
            "output_state": {},
        }

    # Run parser in parallel
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=parser_workers) as executor:
        futures = {executor.submit(parse_one, c): c for c in remaining}
        for future in as_completed(futures):
            result = future.result()
            vname = os.path.basename(result.get("video_path", ""))

            if result.get("parse_success"):
                success_count += 1
                with open(parsed_file, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                sys.stderr.write(f"  ✅ {vname}\n")
            else:
                fail_count += 1
                with open(error_file, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                sys.stderr.write(f"  ❌ {vname}: {result.get('parse_error_type', '')}\n")

    logger.info(f"Parser done: {success_count} success, {fail_count} failed")
    logger.info(f"Parsed outputs: {parsed_file}")


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON object from LLM response text."""
    if not text:
        return None

    # Direct parse
    try:
        return json.loads(text)
    except:
        pass

    # Extract from ```json ... ```
    try:
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return json.loads(text[start:end].strip())
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return json.loads(text[start:end].strip())
    except:
        pass

    # Brace extraction
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except:
        pass

    # json_repair fallback
    try:
        from json_repair import repair_json
        repaired = repair_json(text)
        return json.loads(repaired)
    except:
        pass

    return None


# ======================================================================
# Stage 3: Compare native vs post-hoc
# ======================================================================

def run_comparison(model_name: str, gt_path: str, output_dir: str):
    """
    Compare native MTSS predictions vs post-hoc converted predictions.
    Uses the same metric suite (mtss_metrics.py) for both.
    """
    from .mtss_metrics import evaluate_one_video, load_jsonl

    models_output = output_dir
    caption_dir = os.path.join(output_dir, model_name, "posthoc")
    parsed_file = os.path.join(caption_dir, "posthoc_outputs.jsonl")
    native_file = os.path.join(output_dir, model_name, "results.jsonl")

    if not os.path.exists(parsed_file):
        logger.error(f"No parsed post-hoc outputs at {parsed_file}")
        return
    if not os.path.exists(native_file):
        logger.error(f"No native results at {native_file}")
        return

    # Load GT
    gt_data = load_jsonl(gt_path)
    gt_lookup = {item.get("video_path", ""): item for item in gt_data}

    # Load native predictions
    native_data = load_jsonl(native_file)
    native_lookup = {}
    for item in native_data:
        vp = item.get("video_path", "")
        if vp and item.get("status") == "success":
            native_lookup[vp] = item

    # Load post-hoc predictions
    posthoc_data = load_jsonl(parsed_file)
    posthoc_lookup = {}
    parse_success = 0
    parse_fail = 0
    for item in posthoc_data:
        vp = item.get("video_path", "")
        if vp and item.get("parse_success"):
            posthoc_lookup[vp] = item
            parse_success += 1
        else:
            parse_fail += 1

    # Find common videos
    common_videos = sorted(
        set(native_lookup.keys()) & set(posthoc_lookup.keys()) & set(gt_lookup.keys())
    )
    logger.info(f"Common videos for comparison: {len(common_videos)}")
    logger.info(f"Post-hoc parse: {parse_success} success, {parse_fail} failed")

    if not common_videos:
        logger.error("No common videos to compare!")
        return

    # Evaluate both pipelines
    native_scores = []
    posthoc_scores = []
    comparison_details = []

    for vp in common_videos:
        gt_item = gt_lookup[vp]
        native_item = native_lookup[vp]
        posthoc_item = posthoc_lookup[vp]

        # Build a fake pred_item for post-hoc (same format as native)
        posthoc_pred_item = {
            "video_path": vp,
            "status": "success",
            "model_prediction": posthoc_item.get("output_state", {}),
        }

        try:
            native_result = evaluate_one_video(gt_item, native_item)
            posthoc_result = evaluate_one_video(gt_item, posthoc_pred_item)

            native_scores.append(native_result)
            posthoc_scores.append(posthoc_result)

            comparison_details.append({
                "video_path": vp,
                "native_composite": round(native_result["composite_score"], 4),
                "posthoc_composite": round(posthoc_result["composite_score"], 4),
                "delta": round(native_result["composite_score"] - posthoc_result["composite_score"], 4),
                "native_faa": round(native_result["faa_score"], 4),
                "posthoc_faa": round(posthoc_result["faa_score"], 4),
                "native_ccc": round(native_result["ccc_score"], 4),
                "posthoc_ccc": round(posthoc_result["ccc_score"], 4),
                "native_nhr": round(native_result["nhr_score"], 4),
                "posthoc_nhr": round(posthoc_result["nhr_score"], 4),
                "native_ttoa": round(native_result["ttoa_score"], 4),
                "posthoc_ttoa": round(posthoc_result["ttoa_score"], 4),
            })
        except Exception as e:
            logger.warning(f"Error evaluating {vp}: {e}")

    if not native_scores:
        logger.error("No videos successfully evaluated!")
        return

    # Aggregate
    metric_keys = ["composite_score", "faa_score", "ccc_score", "avar_score",
                    "ttoa_score", "gfid_score", "sgc_score", "nhr_score", "ebc_score",
                    "who_score", "when_score", "where_score", "what_score"]

    results_table = {"native": {}, "posthoc": {}}
    for key in metric_keys:
        native_vals = [r[key] for r in native_scores if key in r]
        posthoc_vals = [r[key] for r in posthoc_scores if key in r]
        if native_vals:
            results_table["native"][key] = round(float(np.mean(native_vals)), 4)
        if posthoc_vals:
            results_table["posthoc"][key] = round(float(np.mean(posthoc_vals)), 4)

    results_table["posthoc"]["parse_success_rate"] = round(
        parse_success / max(parse_success + parse_fail, 1), 4
    )
    results_table["num_compared"] = len(common_videos)

    # Win/loss analysis
    native_wins = sum(1 for d in comparison_details if d["delta"] > 0.01)
    posthoc_wins = sum(1 for d in comparison_details if d["delta"] < -0.01)
    ties = len(comparison_details) - native_wins - posthoc_wins
    results_table["win_loss"] = {
        "native_wins": native_wins,
        "posthoc_wins": posthoc_wins,
        "ties": ties,
    }

    # Save results
    result_dir = os.path.join(caption_dir)
    scores_path = os.path.join(result_dir, "native_vs_posthoc_scores.json")
    with open(scores_path, "w") as f:
        json.dump(results_table, f, indent=2, ensure_ascii=False)

    details_path = os.path.join(result_dir, "native_vs_posthoc_detail.jsonl")
    with open(details_path, "w") as f:
        for d in comparison_details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Print results
    print("\n" + "=" * 70)
    print(f"  Native vs Post-hoc Comparison: {model_name}")
    print("=" * 70)
    print(f"  Videos compared: {len(common_videos)}")
    print(f"  Post-hoc parse success rate: {results_table['posthoc']['parse_success_rate']*100:.1f}%")
    print()
    print(f"  {'Metric':<30s} {'Native':>10s} {'Post-hoc':>10s} {'Delta':>10s}")
    print("  " + "─" * 60)
    for key in metric_keys:
        n_val = results_table["native"].get(key, 0)
        p_val = results_table["posthoc"].get(key, 0)
        delta = n_val - p_val
        marker = "▲" if delta > 0.01 else "▼" if delta < -0.01 else "="
        print(f"  {key:<30s} {n_val:>10.4f} {p_val:>10.4f} {delta:>+10.4f} {marker}")
    print()
    wl = results_table["win_loss"]
    print(f"  Win/Loss: Native {wl['native_wins']} | Post-hoc {wl['posthoc_wins']} | Tie {wl['ties']}")
    print("=" * 70)

    logger.info(f"Results saved to {scores_path}")
    logger.info(f"Details saved to {details_path}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Native vs Post-hoc Experiment")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["caption", "parse", "compare", "all"],
                        help="Which stage to run")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (must have native results already)")
    parser.add_argument("--gt", type=str,
                        default=os.environ.get("OMNICAP_EVAL_INPUT_JSONL", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "eval_sample.jsonl")),
                        help="Path to GT JSONL")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
                        help="Output directory (same as main eval output)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Workers for caption inference")
    parser.add_argument("--indices", type=str, default=None,
                        help="Path to JSON file with list of indices (for multi-node sharding)")
    parser.add_argument("--shard-id", type=int, default=None,
                        help="Shard ID for output file naming (avoids NFS concurrent write corruption)")
    parser.add_argument("--caption-base-url", type=str, default=None,
                        help="Override model base_url for caption stage (e.g. http://localhost:8903/v1)")
    parser.add_argument("--parser-model", type=str, default="Qwen3-VL-235B-Instruct",
                        help="LLM model for parsing captions to MTSS")
    parser.add_argument("--parser-base-url", type=str,
                        default=os.environ.get("OMNICAP_EVAL_PARSER_BASE_URL", "http://localhost:8000/v1"),
                        help="Parser LLM API base URL")
    parser.add_argument("--parser-api-key", type=str, default="EMPTY",
                        help="API key for parser LLM (e.g. 'Bearer xxx')")
    parser.add_argument("--parser-workers", type=int, default=8,
                        help="Parallel workers for LLM parser")
    args = parser.parse_args()

    stages = [args.stage] if args.stage != "all" else ["caption", "parse", "compare"]

    for stage in stages:
        print(f"\n{'='*70}")
        print(f"  Stage: {stage}")
        print(f"{'='*70}\n")

        if stage == "caption":
            run_dense_caption_inference(
                args.model, args.gt, args.output_dir, workers=args.workers,
                caption_base_url=args.caption_base_url,
                indices_file=args.indices, shard_id=args.shard_id,
            )
        elif stage == "parse":
            run_parser(
                args.model, args.output_dir,
                parser_model=args.parser_model,
                parser_base_url=args.parser_base_url,
                parser_workers=args.parser_workers,
                parser_api_key=args.parser_api_key,
            )
        elif stage == "compare":
            run_comparison(args.model, args.gt, args.output_dir)


if __name__ == "__main__":
    main()
