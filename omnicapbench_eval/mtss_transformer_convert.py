#!/usr/bin/env python3
"""
Convert transformer captioner outputs into the MTSS evaluation schema.

The transformer baselines are heterogeneous:
  - asid_caption_3b / asid_caption_7b: plain dense caption text in raw_responses.single_pass.
  - timechat_captioner: JSON segment list in raw_responses.single_pass.
  - avocado / ugc_videocaptioner: usually already MTSS-like JSON, but may have empty/failed rows.

This script preserves the original results.jsonl and writes a converted JSONL that
mtss_metrics.py and mtss_llm_judge.py can consume directly.
"""

import argparse
import copy
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from openai import OpenAI

from .mtss_metrics import as_list, canonicalize_caption, valid_range

LOGGER = logging.getLogger("mtss_transformer_convert")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TRANSFORMER_MODELS = {
    "asid_caption_3b",
    "asid_caption_7b",
    "avocado",
    "timechat_captioner",
    "ugc_videocaptioner",
}

DEFAULT_BASE_URL = os.environ.get("OMNICAP_EVAL_CONVERT_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL_NAME = os.environ.get("OMNICAP_EVAL_CONVERT_MODEL", "gpt-4o")
DEFAULT_TEMPERATURE = float(os.environ.get("OMNICAP_EVAL_CONVERT_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.environ.get("OMNICAP_EVAL_CONVERT_MAX_TOKENS", "32768"))
DEFAULT_MAX_ITERATIONS = int(os.environ.get("OMNICAP_EVAL_CONVERT_MAX_ITERATIONS", "0"))
DEFAULT_API_KEY = os.environ.get("OMNICAP_EVAL_CONVERT_API_KEY", "EMPTY")

SYSTEM_PROMPT = """You are a meticulous converter for video caption outputs.
Convert the provided model output into a valid MTSS JSON object only. Do not add markdown.
The JSON object must have exactly these top-level keys: references, events, shots.
All facts must come from the model output, the given shot boundaries, or obvious timing metadata. Do not invent unsupported identities.
"""

MODEL_NOTES = {
    "asid_caption_3b": "Input is a single dense paragraph with time anchors like 'At 10s'. Extract persistent entities, timed events, and shot-local visual states.",
    "asid_caption_7b": "Input is a single dense paragraph with time anchors like 'At 0s'. Extract persistent entities, timed events, and shot-local visual states.",
    "timechat_captioner": "Input is usually a JSON list of segments with timestamp, segment_detail_caption, camera_state, video_background, speech_content, and acoustics_content.",
    "avocado": "Input is often already MTSS-like. Repair only if model_prediction is empty or malformed; otherwise preserve and sanitize the existing structure.",
    "ugc_videocaptioner": "Input is often already MTSS-like. Repair only if model_prediction is empty or malformed; otherwise preserve and sanitize the existing structure.",
}


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping invalid JSON at %s:%d: %s", path, line_no, exc)
    return rows


def append_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def has_nonempty_mtss(value: Any) -> bool:
    cap, _ = canonicalize_caption(value)
    return any(cap.get(k) for k in ("references", "events", "shots"))


def extract_raw_text(row: dict) -> str:
    raw = row.get("raw_responses")
    if isinstance(raw, dict):
        for key in ("single_pass", "response", "output", "text", "caption"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if raw:
            return json.dumps(raw, ensure_ascii=False)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    pred = row.get("model_prediction")
    if isinstance(pred, str) and pred.strip():
        return pred.strip()
    return ""


def parse_jsonish(text: str) -> Any:
    if not isinstance(text, str) or not text.strip():
        return None
    clean = text.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    start_positions = [p for p in [clean.find("{"), clean.find("[")] if p >= 0]
    if not start_positions:
        return None
    start = min(start_positions)
    end_obj = clean.rfind("}")
    end_arr = clean.rfind("]")
    end = max(end_obj, end_arr)
    if end <= start:
        return None
    try:
        return json.loads(clean[start:end + 1])
    except json.JSONDecodeError:
        return None


def normalize_time(value: Any, default: Optional[List[float]] = None) -> Optional[List[float]]:
    rng = valid_range(value)
    if rng:
        return [float(rng[0]), float(rng[1])]
    if isinstance(value, str):
        nums = re.findall(r"\d+(?:\.\d+)?", value)
        if len(nums) >= 2:
            return [float(nums[0]), float(nums[1])]
        if len(nums) == 1:
            start = float(nums[0])
            return [start, start + 1.0]
    return default


def sanitize_caption(caption: Any, row: dict) -> Dict[str, List[dict]]:
    cap, _ = canonicalize_caption(caption)
    references, events, shots = [], [], []

    for idx, ref in enumerate(as_list(cap.get("references")), 1):
        if not isinstance(ref, dict):
            continue
        item = copy.deepcopy(ref)
        ref_type = str(item.get("type") or "object").lower()
        prefix = {
            "person": "PERSON",
            "animal": "ANIMAL",
            "scene": "SCENE",
            "source": "SOURCE",
            "audio": "AUDIO",
        }.get(ref_type, "OBJECT")
        item["ref_id"] = str(item.get("ref_id") or f"{prefix}_{idx}").upper()
        item["type"] = ref_type
        item["semantic_description"] = str(item.get("semantic_description") or item.get("description") or "")
        anchor = item.get("appearance_anchor") if isinstance(item.get("appearance_anchor"), dict) else {}
        id_features = anchor.get("id_features") if isinstance(anchor.get("id_features"), dict) else {}
        id_features["detail_description"] = str(id_features.get("detail_description") or item.get("detail_description") or item["semantic_description"])
        if "timestamp" not in id_features:
            id_features["timestamp"] = "0.0"
        anchor["id_features"] = id_features
        if not isinstance(anchor.get("attributes"), dict):
            anchor["attributes"] = {}
        item["appearance_anchor"] = anchor
        references.append(item)

    duration = float(row.get("duration_sec") or 0.0)
    for idx, event in enumerate(as_list(cap.get("events")), 1):
        if not isinstance(event, dict):
            continue
        item = copy.deepcopy(event)
        event_type = str(item.get("type") or "visual").lower()
        item["type"] = event_type
        prefix = "DIALOGUE" if event_type == "dialogue" else "EVENT"
        item["event_id"] = str(item.get("event_id") or f"{prefix}_{idx}").upper()
        item["time_range"] = normalize_time(item.get("time_range"), [0.0, duration] if duration else [0.0, 1.0])
        content = item.get("content") if isinstance(item.get("content"), dict) else {}
        desc = content.get("description") or item.get("description") or content.get("line") or ""
        content["description"] = str(desc)
        if event_type == "dialogue" and "line" not in content:
            content["line"] = str(desc)
        item["content"] = content
        events.append(item)

    shot_times = as_list(row.get("shot_time"))
    for idx, shot in enumerate(as_list(cap.get("shots")), 1):
        if not isinstance(shot, dict):
            continue
        item = copy.deepcopy(shot)
        item["shot_id"] = str(item.get("shot_id") or f"SHOT_{idx}").upper()
        fallback = normalize_time(shot_times[idx - 1]) if idx - 1 < len(shot_times) else None
        item["time_range"] = normalize_time(item.get("time_range"), fallback or ([0.0, duration] if duration else [0.0, 1.0]))
        item["shot_transfer"] = str(item.get("shot_transfer") or "cut/unknown")
        camera = item.get("camera") if isinstance(item.get("camera"), dict) else {}
        item["camera"] = {
            "shot_type": str(camera.get("shot_type") or camera.get("type") or "unknown"),
            "angle": str(camera.get("angle") or "unknown"),
            "movement": str(camera.get("movement") or "unknown"),
        }
        subs = []
        for sidx, sub in enumerate(as_list(item.get("visual_description")), 1):
            if not isinstance(sub, dict):
                continue
            sub_item = copy.deepcopy(sub)
            sub_item["sub_shot_id"] = str(sub_item.get("sub_shot_id") or f"{item['shot_id']}_SUB_{sidx}")
            sub_item["sub_shot_time"] = normalize_time(sub_item.get("sub_shot_time"), item["time_range"])
            sub_item["sub_shot_content"] = str(sub_item.get("sub_shot_content") or sub_item.get("description") or "")
            subs.append(sub_item)
        if not subs:
            subs.append({
                "sub_shot_id": f"{item['shot_id']}_SUB_1",
                "sub_shot_time": item["time_range"],
                "sub_shot_content": str(item.get("description") or item.get("summary") or ""),
            })
        item["visual_description"] = subs
        item["active_events"] = as_list(item.get("active_events"))
        item["references_in_shot"] = as_list(item.get("references_in_shot"))
        shots.append(item)

    if not shots and shot_times:
        for idx, shot_time in enumerate(shot_times, 1):
            rng = normalize_time(shot_time, [0.0, duration] if duration else [0.0, 1.0])
            shots.append({
                "shot_id": f"SHOT_{idx}",
                "time_range": rng,
                "shot_transfer": "cut/unknown",
                "camera": {"shot_type": "unknown", "angle": "unknown", "movement": "unknown"},
                "visual_description": [{
                    "sub_shot_id": f"SHOT_{idx}_SUB_1",
                    "sub_shot_time": rng,
                    "sub_shot_content": "",
                }],
                "active_events": [],
                "references_in_shot": [],
            })

    return {"references": references, "events": events, "shots": shots}


def build_conversion_prompt(row: dict, model_key: str, raw_text: str) -> str:
    shot_time = as_list(row.get("shot_time"))
    duration = row.get("duration_sec", 0)
    parsed_raw = parse_jsonish(raw_text)
    raw_block = json.dumps(parsed_raw, ensure_ascii=False, indent=2) if parsed_raw is not None else raw_text
    return f"""
Model key: {model_key}
Model-specific note: {MODEL_NOTES.get(model_key, '')}
Video duration seconds: {duration}
Shot boundaries, use these as the preferred shots and keep their order:
{json.dumps(shot_time, ensure_ascii=False)}

Convert the model output below into MTSS JSON.
Rules:
1. Output only JSON with keys references, events, shots.
2. references: persistent visible people, objects, scenes, animals, audio/source entities. Use IDs PERSON_1, OBJECT_1, SCENE_1, SOURCE_1, AUDIO_1 as appropriate.
3. events: timed actions, dialogue, music, sound effects, text/title-card changes, or important visual state changes. Use EVENT_1... for non-dialogue and DIALOGUE_1... for dialogue. Every event needs event_id, type, time_range, content.description. Dialogue also needs content.line and speaker if inferable.
4. shots: create one shot per provided shot boundary when possible. Every shot needs shot_id, time_range, shot_transfer, camera, visual_description, active_events, references_in_shot.
5. visual_description: include one or more subshots with sub_shot_id, sub_shot_time, sub_shot_content. Use the model output's local visual details.
6. Use only evidence from the model output. If a field is unknown, use an empty list or "unknown" rather than inventing.
7. Keep all time ranges within [0, duration]. If only one timestamp is given, infer a short interval until the next timestamp or shot boundary.

MODEL OUTPUT:
{raw_block}
""".strip()


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str, temperature: float, max_tokens: int, timeout: int, max_retries: int):
        self.client = OpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key if api_key and api_key != "EMPTY" else "EMPTY",
            timeout=timeout,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def complete_json(self, prompt: str) -> Optional[Any]:
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=0.8,
                    presence_penalty=1.5,
                    extra_body={
                        "top_k": 20,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                text = resp.choices[0].message.content or ""
                parsed = parse_jsonish(text)
                if parsed is not None:
                    return parsed
                LOGGER.warning("Could not parse converter response: %s", text[:300])
                return None
            except Exception as exc:
                error_str = str(exc)
                if "429" in error_str or "rate" in error_str.lower():
                    time.sleep(min(2 ** (attempt + 1), 30))
                elif "500" in error_str or "502" in error_str or "503" in error_str:
                    time.sleep(min(2 ** (attempt + 1), 30))
                elif "too large" in error_str or "input_tokens" in error_str:
                    LOGGER.warning("Input too long for converter, skipping: %s", error_str[:200])
                    return None
                else:
                    LOGGER.warning("Request failed attempt %d/%d: %s", attempt + 1, self.max_retries + 1, str(exc)[:200])
                    time.sleep(min(2 * (attempt + 1), 20))
        return None


def should_convert(row: dict, force: bool) -> bool:
    if row.get("status") != "success":
        return False
    if force:
        return True
    return not has_nonempty_mtss(row.get("model_prediction"))


def convert_row(row: dict, model_key: str, client: Optional[OpenAICompatClient], force: bool) -> dict:
    out = copy.deepcopy(row)
    out["conversion_model"] = None
    out["conversion_status"] = "not_needed"

    if row.get("status") != "success":
        out["conversion_status"] = "skipped_failed_inference"
        return out

    if not should_convert(row, force):
        out["model_prediction"] = sanitize_caption(row.get("model_prediction"), row)
        out["conversion_status"] = "preserved_structured"
        return out

    raw_text = extract_raw_text(row)
    if not raw_text:
        out["model_prediction"] = sanitize_caption({}, row)
        out["conversion_status"] = "failed_no_raw_text"
        out["status"] = "failed"
        out["error"] = (out.get("error") or "") + "; conversion failed: no raw text"
        return out

    if client is None:
        out["conversion_status"] = "dry_run_needs_llm"
        return out

    prompt = build_conversion_prompt(row, model_key, raw_text)
    converted = client.complete_json(prompt)
    if converted is None:
        out["model_prediction"] = sanitize_caption({}, row)
        out["conversion_status"] = "failed_llm_conversion"
        out["status"] = "failed"
        out["error"] = (out.get("error") or "") + "; conversion failed: LLM returned invalid JSON"
        return out

    out["model_prediction"] = sanitize_caption(converted, row)
    out["conversion_status"] = "converted_with_llm"
    out["conversion_model"] = client.model
    return out


def existing_video_paths(path: str) -> set:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = row.get("conversion_status", "")
            retryable = status in {"failed_llm_conversion", "dry_run_needs_llm"}
            if row.get("video_path") and not retryable:
                done.add(row["video_path"])
    return done


def infer_model_key(path: str, explicit: str = "") -> str:
    if explicit:
        return explicit
    base = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert transformer caption outputs to MTSS schema")
    parser.add_argument("--input", required=True, help="Source results.jsonl")
    parser.add_argument("--output", default="", help="Converted JSONL path; default: results_mtss_converted.jsonl beside input")
    parser.add_argument("--model-key", default="", help="One of the transformer model keys; inferred from parent dir by default")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
                        help="Maximum rows to process after resume; 0 means all rows")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("OMNICAP_EVAL_CONVERT_WORKERS", "4")))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("OMNICAP_EVAL_CONVERT_TIMEOUT", "180")))
    parser.add_argument("--retries", type=int, default=int(os.environ.get("OMNICAP_EVAL_CONVERT_RETRIES", "2")))
    parser.add_argument("--force", action="store_true", help="Re-convert rows even if model_prediction is already non-empty")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output instead of resuming")
    parser.add_argument("--dry-run", action="store_true", help="Do not call LLM; only report what would be converted")
    args = parser.parse_args()

    model_key = infer_model_key(args.input, args.model_key)
    if model_key not in TRANSFORMER_MODELS:
        LOGGER.warning("Model key %s is not in known transformer model list; continuing with generic conversion", model_key)

    output = args.output or os.path.join(os.path.dirname(args.input), "results_mtss_converted.jsonl")
    if args.overwrite and os.path.exists(output):
        os.remove(output)

    rows = load_jsonl(args.input)
    done = existing_video_paths(output)
    pending = [r for r in rows if r.get("video_path") not in done]
    if args.max_iterations > 0:
        pending = pending[:args.max_iterations]

    LOGGER.info("Input rows: %d, already converted: %d, pending this run: %d", len(rows), len(done), len(pending))
    client = None if args.dry_run else OpenAICompatClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.retries,
    )

    stats = {"converted_with_llm": 0, "preserved_structured": 0, "skipped_failed_inference": 0,
             "failed_no_raw_text": 0, "failed_llm_conversion": 0, "dry_run_needs_llm": 0}

    def work(row: dict) -> dict:
        return convert_row(row, model_key, client, args.force)

    batch = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(work, row): row for row in pending}
        for idx, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            stats[result.get("conversion_status", "unknown")] = stats.get(result.get("conversion_status", "unknown"), 0) + 1
            batch.append(result)
            if len(batch) >= 20:
                append_jsonl(output, batch)
                batch = []
            if idx % 20 == 0 or idx == len(pending):
                LOGGER.info("Progress %d/%d stats=%s", idx, len(pending), stats)
    if batch:
        append_jsonl(output, batch)

    LOGGER.info("Converted output saved to %s", output)
    LOGGER.info("Final stats: %s", stats)


if __name__ == "__main__":
    main()
