#!/usr/bin/env python3
"""MTSS Eval - Inference Engine
Core inference logic supporting:
  1. Single-pass inference (short videos / closed-source models)
     Uses system_prompt.txt to generate all (references + events + shots) in one call.
  2. Two-pass inference (long videos with qwen3_omni)
     - Pass 1: full video → reference_bank (ref_system_prompt.txt)
     - Pass 2: shot groups (~60s each) → event + shot per group (event_shot_system_prompt.txt)

Content building differs by model family:
  - closed_source (gemini/qwen35): video_url (HTTP URL) + prompt
  - qwen3_omni / qwen25_omni: base64 video + base64 audio + prompt
"""
import json
import logging
import os
import tempfile
import uuid

from . import eval_config as config
from . import video_utils
from .model_client import INPUT_TOO_LONG_SENTINEL

logger = logging.getLogger("mtss_eval")

# ======================================================================
# Shared model client cache (avoid reloading model for every video)
# ======================================================================

_client_cache = {}


def _get_shared_client(model_cfg: dict, tag: str):
    """Return a cached model client instance. For Transformers models this avoids
    reloading the model weights for every single video."""
    # Include sampling params in cache key to prevent cache pollution
    temp = model_cfg.get("temperature", "default")
    top_p = model_cfg.get("top_p", "default")
    cache_key = f"{tag}_t{temp}_p{top_p}"
    if cache_key not in _client_cache:
        from .model_client import create_model_client
        _client_cache[cache_key] = create_model_client(model_cfg, tag=tag)
    return _client_cache[cache_key]


# ======================================================================
# Prompt loading
# ======================================================================

_prompt_cache = {}


def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompt directory."""
    if prompt_name in _prompt_cache:
        return _prompt_cache[prompt_name]
    path = os.path.join(config.PROMPT_DIR, prompt_name)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    _prompt_cache[prompt_name] = text
    return text


# ======================================================================
# Content building
# ======================================================================

def build_content_for_closed_source(video_http_url: str, text_prompt: str) -> list:
    """
    Build content list for closed-source models (Gemini, Qwen3.5-Omni-Plus).
    Uses HTTP video URL directly (no base64, no separate audio).
    """
    content = [
        {
            "type": "video_url",
            "video_url": {
                "url": video_http_url,
                "_http_url": video_http_url,
            }
        },
        {"type": "text", "text": text_prompt},
    ]
    return content


def build_content_for_transformers(video_path: str, text_prompt: str) -> list:
    """
    Build content list for Transformers native models.
    Passes the raw video file path directly — no base64 encoding needed.
    The Qwen25OmniTransformersClient will use qwen_omni_utils.process_mm_info
    to extract audio/video from the file path.
    """
    content = []
    content.append({"type": "video", "video": video_path})
    if text_prompt:
        content.append({"type": "text", "text": text_prompt})
    return content


def build_content_for_qwen_omni(video_path: str, text_prompt: str,
                                 audio_path: str = None,
                                 cache_dir: str = None) -> list:
    """
    Build content list for Qwen-Omni models (base64 video + audio).
    """
    content = []

    # Video as base64
    video_url = video_utils.build_video_data_url(video_path)
    content.append({"type": "video_url", "video_url": {"url": video_url}})

    # Audio as base64
    if audio_path is None:
        audio_path = video_utils.extract_audio(video_path, cache_dir=cache_dir)
    if audio_path and os.path.exists(audio_path):
        audio_url = video_utils.build_audio_data_url(audio_path)
        content.append({"type": "audio_url", "audio_url": {"url": audio_url}})
        # Clean up extracted audio file immediately after encoding
        try:
            os.remove(audio_path)
        except OSError:
            pass

    # Text prompt
    content.append({"type": "text", "text": text_prompt})

    return content


# ======================================================================
# Single-pass inference (all 3 rounds in sequence)
# ======================================================================

def run_single_pass(client, video_item: dict, model_family: str,
                    processed_video_path: str = None,
                    cache_dir: str = None,
                    model_cfg: dict = None) -> dict:
    """
    Run single-pass inference using system_prompt.txt.
    One model call generates the complete MTSS (references + events + shots).
    For short videos (<1min) or closed-source models with large context.

    Args:
        client: model client instance
        video_item: dict from input JSONL
        model_family: "closed_source" / "qwen3_omni" / "qwen25_omni"
        processed_video_path: pre-processed video path (compressed+timestamp)
        cache_dir: directory for temp files

    Returns:
        dict with keys: references, events, shots, raw_responses
    """
    video_path = video_utils.resolve_media_path(video_item)
    video_url = video_item.get("video_url", "")
    shot_time = video_item.get("shot_time", [])
    duration = video_item.get("duration_sec", 0)

    result = {
        "references": None,
        "events": None,
        "shots": None,
        "raw_responses": {},
        "status": "success",
        "error": None,
    }

    # ---- Single call: system_prompt.txt → full MTSS ----
    system_prompt = load_prompt("system_prompt.txt")
    # Inject shot_time into the prompt so model strictly follows pre-defined shot boundaries
    system_prompt = system_prompt.replace(
        "{shot_time}", json.dumps(shot_time, ensure_ascii=False)
    )

    if model_family == "closed_source":
        content = build_content_for_closed_source(video_url, "")
    elif model_cfg and model_cfg.get("inference_backend") == "transformers":
        # For Transformers native models, pass the video file path directly
        # (no need to base64-encode and decode back — wasteful for local inference)
        if not processed_video_path or not os.path.exists(processed_video_path):
            result["status"] = "failed"
            result["error"] = f"single_pass failed: preprocessed video missing: {processed_video_path}"
            return result
        content = build_content_for_transformers(processed_video_path, "")
    else:
        if not processed_video_path or not os.path.exists(processed_video_path):
            result["status"] = "failed"
            result["error"] = f"single_pass failed: preprocessed video missing: {processed_video_path}"
            return result
        content = build_content_for_qwen_omni(processed_video_path, "", cache_dir=cache_dir)

    logger.info(f"[SinglePass] One-shot full MTSS for {os.path.basename(video_path)}")
    response = client.inference(content, system_prompt=system_prompt)

    if not response or response == INPUT_TOO_LONG_SENTINEL:
        result["status"] = "failed"
        result["error"] = f"single_pass failed: {'input_too_long' if response == INPUT_TOO_LONG_SENTINEL else 'empty_response'}"
        return result

    result["raw_responses"]["single_pass"] = response

    # Parse full MTSS response
    try:
        parsed = _parse_json_response(response)
        result["references"] = parsed.get("references", [])
        result["events"] = parsed.get("events", [])
        result["shots"] = parsed.get("shots", [])
        # Also capture top-level fields if present
        for key in ["scene_description", "global_style", "global_audio"]:
            if key in parsed:
                result[key] = parsed[key]
    except Exception as e:
        result["status"] = "failed"
        result["error"] = f"single_pass parse error: {e}"
        return result

    # ---- Sanity check: detect repetition loops / garbage output ----
    n_refs = len(result["references"] or [])
    n_events = len(result["events"] or [])
    n_shots_out = len(result["shots"] or [])
    n_shots_gt = len(shot_time) if shot_time else 1

    # Case 1: parsed OK but both events and shots are empty
    if n_refs > 0 and n_events == 0 and n_shots_out == 0:
        result["status"] = "failed"
        result["error"] = (f"single_pass sanity_check failed: refs={n_refs} but "
                           f"events=0 shots=0 (likely repetition loop)")
        return result

    # Case 2: absurd number of references (>5x expected shots)
    if n_refs > max(n_shots_gt * 5, 30):
        result["status"] = "failed"
        result["error"] = (f"single_pass sanity_check failed: refs={n_refs} is absurdly high "
                           f"(expected ~{n_shots_gt} shots, likely repetition loop)")
        return result

    # Case 3: absurd number of output shots (>3x GT shots)
    if n_shots_out > max(n_shots_gt * 3, 20):
        result["status"] = "failed"
        result["error"] = (f"single_pass sanity_check failed: output shots={n_shots_out} is absurdly high "
                           f"(expected ~{n_shots_gt} shots, likely repetition loop)")
        return result

    return result


# ======================================================================
# Two-pass inference (for long videos with qwen3_omni)
# ======================================================================

def run_two_pass(client, video_item: dict, model_family: str,
                 processed_video_path: str = None,
                 cache_dir: str = None,
                 model_cfg: dict = None) -> dict:
    """
    Run two-pass inference for long videos (>1min) with qwen3_omni:
      Pass 1: full video (fps=2, max_frames=192) → reference_bank
      Pass 2: for each shot group (~60s), cut video segment → event + shot

    Args:
        client: model client instance
        video_item: dict from input JSONL
        model_family: should be "qwen3_omni"
        processed_video_path: pre-processed full video path
        cache_dir: directory for temp files
        model_cfg: model config dict (needed to determine local vs API)

    Returns:
        dict with keys: references, events, shots, raw_responses, shot_groups
    """
    video_path = video_utils.resolve_media_path(video_item)
    video_url = video_item.get("video_url", "")
    shot_time = video_item.get("shot_time", [])
    duration = video_item.get("duration_sec", 0)

    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="mtss_eval_")

    result = {
        "references": None,
        "events": [],
        "shots": [],
        "raw_responses": {},
        "shot_groups": [],
        "status": "success",
        "error": None,
    }

    # ---- Pass 1: Reference Bank (full video) ----
    ref_prompt = load_prompt("ref_system_prompt.txt")
    content = build_content_for_qwen_omni(processed_video_path, "", cache_dir=cache_dir)

    logger.info(f"[TwoPass] Pass 1: reference_bank for {os.path.basename(video_path)}")
    ref_response = client.inference(content, system_prompt=ref_prompt)

    if not ref_response or ref_response == INPUT_TOO_LONG_SENTINEL:
        reason = "input_too_long" if ref_response == INPUT_TOO_LONG_SENTINEL else "empty_response"
        result["status"] = "failed"
        result["error"] = f"Pass1 reference_bank failed: {reason}"
        return result

    result["raw_responses"]["pass1_reference_bank"] = ref_response

    try:
        ref_data = _parse_json_response(ref_response)
        references = ref_data.get("references", [])
        result["references"] = references
    except Exception as e:
        result["status"] = "failed"
        result["error"] = f"Pass1 reference_bank parse error: {e}"
        return result

    # ---- Pass 2: Shot groups → event + shot per group (single call each) ----
    shot_groups = video_utils.group_shots_by_duration(
        shot_time, config.SHOT_GROUP_TARGET_DURATION
    )
    result["shot_groups"] = [[g[0][0], g[-1][1]] for g in shot_groups]

    logger.info(f"[TwoPass] Pass 2: {len(shot_groups)} shot groups for {os.path.basename(video_path)}")

    all_events = []
    all_shots = []
    raw_group_responses = {}

    for gi, group in enumerate(shot_groups):
        group_start = group[0][0]
        group_end = group[-1][1]
        group_shot_time = group

        logger.info(f"[TwoPass] Group {gi+1}/{len(shot_groups)}: {group_start:.1f}s - {group_end:.1f}s")

        # Cut video segment for this group
        segment_path = os.path.join(cache_dir, f"segment_g{gi}_{uuid.uuid4().hex[:6]}.mp4")
        if not video_utils.extract_video_segment(processed_video_path, segment_path,
                                                  group_start, group_end):
            logger.warning(f"[TwoPass] Failed to extract segment for group {gi+1}")
            continue

        # Preprocess segment (compress + timestamp)
        segment_processed = video_utils.preprocess_video(
            segment_path,
            resolution=config.SAMPLING_STRATEGY[model_family]["resolution"],
            add_timestamp=True,
            cache_dir=cache_dir,
        )

        # Adjust shot_time to be relative to segment start
        relative_shot_time = [
            [round(s[0] - group_start, 2), round(s[1] - group_start, 2)]
            for s in group_shot_time
        ]

        # ---- Single call: event_shot_system_prompt.txt → events + shots for this group ----
        es_prompt_template = load_prompt("event_shot_system_prompt.txt")
        es_prompt = es_prompt_template.replace(
            "{references}", json.dumps(references, ensure_ascii=False, indent=2)
        ).replace(
            "{shot_time}", json.dumps(relative_shot_time, ensure_ascii=False)
        )

        content = build_content_for_qwen_omni(segment_processed, "", cache_dir=cache_dir)
        es_response = client.inference(content, system_prompt=es_prompt)

        raw_group_responses[f"group{gi}_event_shot"] = es_response

        group_events = []
        group_shots = []
        if es_response and es_response != INPUT_TOO_LONG_SENTINEL:
            try:
                parsed = _parse_json_response(es_response)
                group_events = parsed.get("events", [])
                group_shots = parsed.get("shots", [])

                # Adjust event time_range back to absolute time
                for ev in group_events:
                    if "time_range" in ev:
                        ev["time_range"] = [
                            round(ev["time_range"][0] + group_start, 2),
                            round(ev["time_range"][1] + group_start, 2),
                        ]
                all_events.extend(group_events)

                # Adjust shot time_range and sub_shot_time back to absolute time
                for shot in group_shots:
                    if "time_range" in shot:
                        shot["time_range"] = [
                            round(shot["time_range"][0] + group_start, 2),
                            round(shot["time_range"][1] + group_start, 2),
                        ]
                    if "visual_description" in shot and isinstance(shot["visual_description"], list):
                        for sub in shot["visual_description"]:
                            if "sub_shot_time" in sub:
                                sub["sub_shot_time"] = [
                                    round(sub["sub_shot_time"][0] + group_start, 2),
                                    round(sub["sub_shot_time"][1] + group_start, 2),
                                ]
                # Renumber shot_ids to be globally unique
                base_shot_idx = len(all_shots)
                for si, shot in enumerate(group_shots):
                    shot["shot_id"] = f"shot_{base_shot_idx + si + 1}"
                all_shots.extend(group_shots)
            except Exception as e:
                logger.warning(f"[TwoPass] Group {gi+1} event_shot parse error: {e}")

        # Clean up segment
        for p in [segment_path, segment_processed]:
            if p and os.path.exists(p) and p != processed_video_path:
                try:
                    os.remove(p)
                except Exception:
                    pass

    result["events"] = all_events
    result["shots"] = all_shots
    result["raw_responses"].update(raw_group_responses)

    # Re-number event_ids globally
    for i, ev in enumerate(result["events"]):
        ev_type = ev.get("type", "unknown").upper()
        ev["event_id"] = f"{ev_type}_{i+1}"

    return result


# ======================================================================
# Main inference dispatcher
# ======================================================================

def run_inference(model_cfg: dict, video_item: dict,
                  cache_dir: str = None) -> dict:
    """
    Main entry point: dispatch to single-pass or two-pass based on model family and video duration.

    Args:
        model_cfg: model preset dict (must include model_family)
        video_item: dict from input JSONL
        cache_dir: directory for temp files

    Returns:
        dict with inference results
    """
    model_family = model_cfg.get("model_family", "closed_source")
    strategy = config.SAMPLING_STRATEGY.get(model_family, config.SAMPLING_STRATEGY["closed_source"])
    duration = video_item.get("duration_sec", 0)
    video_path = video_utils.resolve_media_path(video_item)

    # Check if we should skip this video
    if strategy.get("skip_long_video") and duration > strategy.get("max_duration", 60):
        return {
            "status": "skipped",
            "error": f"Video duration {duration:.1f}s exceeds max {strategy.get('max_duration', 60)}s for {model_family}",
        }

    # Reuse a shared model client across videos (avoids reloading model each time)
    tag = f"{model_cfg.get('model_type', 'unknown')}_{model_family}"
    client = _get_shared_client(model_cfg, tag)

    # Preprocess video (compress to 448x448 + timestamp)
    processed_video_path = None
    if model_family != "closed_source":
        if cache_dir is None:
            cache_dir = tempfile.mkdtemp(prefix="mtss_eval_")
        processed_video_path = video_utils.preprocess_video(
            video_path,
            resolution=strategy["resolution"],
            add_timestamp=True,
            cache_dir=cache_dir,
        )

    # Decide single-pass vs two-pass
    two_pass_threshold = strategy.get("two_pass_threshold")
    if two_pass_threshold and duration > two_pass_threshold:
        logger.info(f"[Dispatch] Two-pass inference for {os.path.basename(video_path)} "
                     f"(duration={duration:.1f}s > threshold={two_pass_threshold}s)")
        return run_two_pass(client, video_item, model_family,
                            processed_video_path, cache_dir,
                            model_cfg=model_cfg)
    else:
        logger.info(f"[Dispatch] Single-pass inference for {os.path.basename(video_path)} "
                     f"(duration={duration:.1f}s)")
        return run_single_pass(client, video_item, model_family,
                               processed_video_path, cache_dir,
                               model_cfg=model_cfg)


# ======================================================================
# JSON parsing utility
# ======================================================================

def _parse_json_response(response: str) -> dict:
    """Parse model response text into a Python dict."""
    if not response:
        return {}

    def _ensure_dict(obj):
        if isinstance(obj, list):
            if obj and isinstance(obj[0], dict):
                if "ref_id" in obj[0]:
                    return {"references": obj}
                elif "event_id" in obj[0]:
                    return {"events": obj}
                elif "shot_id" in obj[0]:
                    return {"shots": obj}
            return {"items": obj}
        if isinstance(obj, dict):
            return obj
        return {"raw_response": str(obj)}

    # 1. Direct parse
    try:
        return _ensure_dict(json.loads(response))
    except json.JSONDecodeError:
        pass

    # 2. Extract from ```json ... ```
    try:
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            return _ensure_dict(json.loads(response[start:end].strip()))
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            return _ensure_dict(json.loads(response[start:end].strip()))
    except Exception:
        pass

    # 3. json_repair
    try:
        from json_repair import repair_json
        repaired = repair_json(response)
        return _ensure_dict(json.loads(repaired))
    except Exception:
        pass

    # 4. Brace extraction
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            return _ensure_dict(json.loads(response[start:end]))
    except Exception:
        pass

    # 5. Array extraction
    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end > start:
            return _ensure_dict(json.loads(response[start:end]))
    except Exception:
        pass

    return {"raw_response": response, "parse_error": True}
