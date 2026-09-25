#!/usr/bin/env python3
"""
MTSS Pipeline - Main Pipeline Orchestrator
Orchestrates 3 rounds (reference_bank -> event -> shot) x 4 stages (gen -> eval -> refine -> select)
complete workflow, including feedback loop, retry, and fallback control logic.

Workflow Overview:
  Round 1 (reference_bank): audio+video+prompt -> references
  Round 2 (event):          audio+video+prompt+references -> events
  Round 3 (shot):           audio+video+prompt+references+events -> shots

  Each round has 4 Stages:
    Stage1 Gen    -> Generate content (1 model)
    Stage2 Eval   -> Factual accuracy verification (2-3 models vote)
    Stage3 Refine -> Description quality improvement (2 models)
    Stage4 Select -> Atomic unit selection and concatenation (1 model)

  Stage2 voting logic (supports 2 or 3 evaluation models):
    All pass -> Stage3
    Partial failure (but not all fail) -> feedback, fall back to Stage1 retry
    All fail -> discard, restart Stage1

  Stage3 iteration logic:
    Both refined captions pass Stage2 -> Stage4
    Some fail -> fall back to Stage1, regenerate then refine
    max_turn reached:
      1 passes -> use the passing one
      Neither passes -> use the unrefined version that passed Stage2
"""

import json
import os
import logging
import time
from datetime import datetime
from typing import Optional

from . import config
from .model_client import BaseModelClient, ModelClient, create_model_client
from .media_utils import (
    detect_shots, extract_audio_from_video, build_multimodal_content,
    compress_video, add_timestamp_to_video, build_video_url, build_audio_url,
)
from .stages.stage1_gen import run_stage1
from .stages.stage2_eval import run_stage2, run_stage2_single
from .stages.stage3_refine import run_stage3
from .stages.stage4_select import run_stage4
from .validators import (
    validate_stage1_output, format_validation_feedback, get_video_duration,
    fix_sub_shot_times, auto_fix_structural_issues,
)

import tempfile
import uuid

logger = logging.getLogger("omnicapbench_pipeline")
summary_logger = logging.getLogger("omnicapbench_pipeline.summary")


class PipelineRejectError(Exception):
    """Raised when a pipeline item fails due to validation rejection (not API/network errors).
    This means all Stage1-Stage2 attempts were exhausted and the content could not pass evaluation.
    These items can be safely skipped with --skip-failed since retrying won't help.
    """
    pass


# ======================================================================
# Intermediate result recorder for debugging and process optimization
# ======================================================================

class IntermediateRecorder:
    """
    Records intermediate results from each stage to disk for debugging and workflow analysis.
    Directory structure:
      debug/<DID>/
        round1_reference_bank/
          stage1_gen_attempt1.json
          stage1_gen_attempt2.json
          stage2_eval_attempt1.json        # Contains full evaluations from all 3 models
          stage2_eval_attempt2.json
          stage2_verified.json             # Content that passed Stage2
          stage3_refine_a.json
          stage3_refine_b.json
          stage3_reeval_attempt1.json
          stage4_candidate_a.json
          stage4_candidate_b.json
          stage4_final.json
        round2_event/
          ...
        round3_shot/
          ...
        final_mtss.json                    # Final complete result
    """

    def __init__(self, did: str, base_dir: str = None):
        self.did = did
        self.enabled = config.SAVE_INTERMEDIATE
        if self.enabled:
            base = base_dir or config.DEBUG_DIR
            self.base_path = os.path.join(base, self._safe_name(did))
            os.makedirs(self.base_path, exist_ok=True)
        else:
            self.base_path = None

    @staticmethod
    def _safe_name(name: str) -> str:
        """Convert DID to a safe directory name."""
        return name.replace("/", "_").replace("\\", "_").replace(" ", "_")

    def _round_dir(self, round_name: str, round_idx: int) -> str:
        """Get the directory path for a given round."""
        d = os.path.join(self.base_path, f"round{round_idx}_{round_name}")
        os.makedirs(d, exist_ok=True)
        return d

    def save(self, round_name: str, round_idx: int,
             filename: str, data: dict) -> None:
        """Save intermediate result to a JSON file."""
        if not self.enabled:
            return
        try:
            d = self._round_dir(round_name, round_idx)
            path = os.path.join(d, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"[Recorder] Saved {path}")
        except Exception as e:
            logger.warning(f"[Recorder] Failed to save {filename}: {e}")

    def save_final(self, data: dict) -> None:
        """Save the final complete MTSS result."""
        if not self.enabled:
            return
        try:
            path = os.path.join(self.base_path, "final_mtss.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"[Recorder] Saved final: {path}")
        except Exception as e:
            logger.warning(f"[Recorder] Failed to save final: {e}")

    def save_raw_response(self, round_name: str, round_idx: int,
                          filename: str, raw_text: str) -> None:
        """Save the model's raw text response (for debugging JSON parse issues)."""
        if not self.enabled:
            return
        try:
            d = self._round_dir(round_name, round_idx)
            path = os.path.join(d, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(raw_text)
            logger.debug(f"[Recorder] Saved raw: {path}")
        except Exception as e:
            logger.warning(f"[Recorder] Failed to save raw {filename}: {e}")


# ======================================================================
# Video/audio preprocessing cache (avoids redundant processing when the
# same video is used multiple times within a round)
# ======================================================================

class MediaCache:
    """
    Caches preprocessed video and audio content components to avoid redundant
    compression/timestamp overlay/base64 encoding for the same video.
    Each video is processed only once; subsequent calls reuse the cached content list.
    Also tracks created temporary files and cleans them up on clear().

    For models like Gemini / Qwen3.5-Omni-Plus that require HTTP URLs,
    set_video_http_url() can be used to provide the video's COS/HTTP URL.
    build_content_with_text() will automatically include that URL in the content.
    """

    def __init__(self):
        self._cache = {}  # key: (video_path, audio_path) -> list of content dicts
        self._temp_files = []  # Track all created temp files for cleanup
        self._video_http_url = None  # COS/HTTP URL for Gemini/Qwen3.5-Omni-Plus

    def set_video_http_url(self, url: str):
        """
        Set the HTTP URL for the video (e.g., COS signed URL from input data's url_t field).
        This URL will be used by Gemini/Qwen3.5-Omni-Plus models instead of base64.
        """
        self._video_http_url = url
        if url:
            logger.info(f"[MediaCache] Video HTTP URL set: {url[:80]}...")

    def get_media_content(self, video_path: str, audio_path: str) -> list:
        """
        Get the preprocessed video + audio content list (without text prompt).

        If _video_http_url is set, ALL models use the HTTP URL directly
        (no local compression / timestamp overlay / base64 encoding needed).
        This ensures every model sees the exact same video from the COS URL.

        If _video_http_url is NOT set, fall back to the legacy path:
        compress + timestamp overlay + base64 encode the local file.
        """
        key = (video_path, audio_path)
        if key in self._cache:
            logger.debug(f"[MediaCache] HIT for {video_path}")
            return self._cache[key]

        content = []

        if self._video_http_url:
            # ---- HTTP URL mode: skip all local preprocessing ----
            # All models (Gemini, Qwen3.5-Omni-Plus) will use this URL directly.
            logger.info(
                f"[MediaCache] HTTP URL mode — skipping local preprocessing, "
                f"using {self._video_http_url[:80]}..."
            )
            content.append({
                "type": "video_url",
                "video_url": {
                    "url": self._video_http_url,       # primary URL for all models
                    "_http_url": self._video_http_url,  # explicit HTTP URL field
                },
            })
        else:
            # ---- Legacy base64 mode: compress + timestamp + encode ----
            logger.info(f"[MediaCache] MISS, preprocessing {video_path}")

            tmp = os.path.join(tempfile.gettempdir(),
                               f"compress_{uuid.uuid4().hex[:8]}.mp4")
            self._temp_files.append(tmp)
            if compress_video(video_path, tmp, 720):
                vp = add_timestamp_to_video(tmp)
            else:
                vp = add_timestamp_to_video(video_path)
            if vp != video_path and vp != tmp:
                self._temp_files.append(vp)

            video_url = build_video_url(vp)
            content.append({"type": "video_url", "video_url": {"url": video_url}})

            # --- Audio (only needed for local QwenOmni which accepts base64 audio) ---
            ap = audio_path
            if not ap or not os.path.exists(ap):
                ap = extract_audio_from_video(video_path)
            if ap and os.path.exists(ap):
                if ap != audio_path:
                    self._temp_files.append(ap)
                audio_url = build_audio_url(ap)
                content.append({"type": "audio_url", "audio_url": {"url": audio_url}})

        self._cache[key] = content
        return content

    def build_content_with_text(self, video_path: str, audio_path: str,
                                text_prompt: str) -> list:
        """
        Get complete multimodal content (video + audio + text).

        When _video_http_url is set, get_media_content() already produces
        content with the HTTP URL — no further patching needed.
        When _video_http_url is NOT set (legacy base64 mode), content is
        returned as-is with base64 data URLs.
        """
        media = self.get_media_content(video_path, audio_path)
        result = list(media)  # return a copy to avoid mutating the cache
        result.append({"type": "text", "text": text_prompt})
        return result

    def clear(self):
        """Clear cache and clean up all temporary files."""
        self._cache.clear()
        for f in self._temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    logger.debug(f"[MediaCache] Cleaned up temp file: {f}")
            except Exception as e:
                logger.warning(f"[MediaCache] Failed to clean up temp file {f}: {e}")
        logger.info(f"[MediaCache] Cleaned up {len(self._temp_files)} temp files")
        self._temp_files.clear()


# ======================================================================
# Model initialization
# ======================================================================

def _init_all_models():
    """
    Initialize model clients for all stages.
    Uses the create_model_client factory function, which auto-selects the backend
    based on the model_type in config.
    """
    models = {
        "stage1": create_model_client(config.STAGE1_MODEL, "stage1_gen"),
        "stage2": [
            create_model_client(cfg, f"stage2_eval_{i}")
            for i, cfg in enumerate(config.STAGE2_MODELS)
        ],
        "stage3": [
            create_model_client(cfg, f"stage3_refine_{i}")
            for i, cfg in enumerate(config.STAGE3_MODELS)
        ],
        "stage4": create_model_client(config.STAGE4_MODEL, "stage4_select"),
    }
    logger.info(
        f"Models initialized: "
        f"stage1={models['stage1']}, "
        f"stage2={[str(m) for m in models['stage2']]}, "
        f"stage3={[str(m) for m in models['stage3']]}, "
        f"stage4={models['stage4']}"
    )
    return models


# ======================================================================
# Shot chunking helper: filter events by time range
# ======================================================================

def _filter_events_for_time_range(events: list, start_time: float, end_time: float,
                                   tolerance: float = 3.0) -> list:
    """
    Filter events that overlap with the given time range [start_time, end_time].
    Used by shot chunking to pass only relevant events to each chunk.

    Args:
        events:     Full events list from Round 2
        start_time: Chunk start time (seconds)
        end_time:   Chunk end time (seconds)
        tolerance:  Time tolerance for overlap detection (seconds)

    Returns:
        Filtered events list (preserving original event_ids and structure)
    """
    if not events or not isinstance(events, list):
        return events

    filtered = []
    for evt in events:
        if not isinstance(evt, dict):
            filtered.append(evt)
            continue
        tr = evt.get("time_range", [])
        if not isinstance(tr, list) or len(tr) != 2:
            filtered.append(evt)  # Keep events with invalid time_range
            continue
        evt_start, evt_end = tr
        if not isinstance(evt_start, (int, float)) or not isinstance(evt_end, (int, float)):
            filtered.append(evt)
            continue
        # Check overlap with tolerance
        if evt_end + tolerance >= start_time and evt_start - tolerance <= end_time:
            filtered.append(evt)

    logger.debug(
        f"[Shot-Chunking] Filtered events for [{start_time}, {end_time}]: "
        f"{len(filtered)}/{len(events)} events"
    )
    return filtered


# ======================================================================
# Single round pipeline (complete workflow of 4 stages)
# ======================================================================

def _run_single_round(
    models: dict,
    round_name: str,
    video_path: str,
    audio_path: str,
    shot_timestamps: list,
    media_cache: MediaCache,
    references: dict = None,
    events: dict = None,
    recorder: IntermediateRecorder = None,
    round_idx: int = 1,
) -> Optional[dict]:
    """
    Execute a complete round (4 stages).

    Args:
        models:          Dictionary of all model clients
        round_name:      "reference_bank" | "event" | "shot"
        video_path:      Path to the video file
        audio_path:      Path to the audio file
        shot_timestamps: List of shot boundaries
        media_cache:     Media preprocessing cache
        references:      Output from the reference_bank round
        events:          Output from the event round

    Returns:
        The final output dict for this round, or None on failure.
    """
    round_start = time.time()
    logger.info(f"\n{'='*60}")
    logger.info(f"  ROUND: {round_name}")
    logger.info(f"{'='*60}")
    print(f"\n\U0001f504 Starting round: {round_name}")

    # Get video duration for validation bounds checking
    video_duration = get_video_duration(video_path)
    logger.info(f"[Round:{round_name}] video_duration={video_duration}s")

    # ================================================================
    # Stage1 + Stage2 loop
    # ================================================================
    verified_content = None
    feedback = None

    for attempt in range(config.STAGE2_MAX_RETRY + 1):
        # --- Stage1: Generation ---
        print(f"  \U0001f4dd Stage1 Gen (attempt {attempt + 1}/{config.STAGE2_MAX_RETRY + 1})...")
        logger.info(f"[Round:{round_name}] Stage1 attempt {attempt + 1}")

        generated = run_stage1(
            model=models["stage1"],
            round_name=round_name,
            video_path=video_path,
            audio_path=audio_path,
            shot_timestamps=shot_timestamps,
            references=references,
            events=events,
            feedback=feedback,
            media_cache=media_cache,
        )

        # Record Stage1 generation result
        if recorder:
            recorder.save(round_name, round_idx,
                          f"stage1_gen_attempt{attempt+1}.json",
                          {"feedback": feedback, "generated": generated})

        if generated.get("parse_error"):
            logger.warning(f"[Round:{round_name}] Stage1 JSON parse failed, retrying...")
            print(f"  \u26a0\ufe0f  Stage1 JSON parse failed, retrying...")
            feedback = "Previous generation failed to produce valid JSON. Please output valid JSON only, with no extra text."
            continue

        # Validate output contains the correct key (prevent round mismatch)
        expected_key_map = {
            "reference_bank": "references",
            "event": "events",
            "shot": "shots",
        }
        expected_key = expected_key_map.get(round_name)
        if expected_key and expected_key not in generated:
            logger.warning(
                f"[Round:{round_name}] Stage1 output missing expected '{expected_key}' key, "
                f"got keys={list(generated.keys())}"
            )
            print(f"  \u26a0\ufe0f  Stage1 output missing '{expected_key}' key, retrying...")
            feedback = (
                f"Your output is missing the required '{expected_key}' key. "
                f"You output keys: {list(generated.keys())}. "
                f"Please output a JSON object containing only the '{expected_key}' array."
            )
            continue

        # --- Programmatic structural validation (BEFORE Stage2) ---
        # Step 1: Auto-fix minor structural issues that can be corrected programmatically
        print(f"  \U0001f527 Auto-fixing structural issues...")
        auto_fixes = auto_fix_structural_issues(
            round_name=round_name,
            generated=generated,
            shot_timestamps=shot_timestamps,
            references=references,
            events=events,
            video_duration=video_duration,
        )
        if auto_fixes:
            print(f"  \u2705 Auto-fixed {len(auto_fixes)} structural issues")
            for af in auto_fixes[:5]:  # Show first 5 fixes
                print(f"      \u2714 {af}")
            if len(auto_fixes) > 5:
                print(f"      ... and {len(auto_fixes) - 5} more")
        else:
            print(f"  \u2705 No structural issues to auto-fix")

        # Record auto-fix results
        if recorder:
            recorder.save(round_name, round_idx,
                          f"auto_fix_attempt{attempt+1}.json",
                          {"fixes": auto_fixes, "fix_count": len(auto_fixes),
                           "content_after_fix": generated})

        # Step 2: Validate remaining structural issues (after auto-fix)
        print(f"  \U0001f50e Structural validation...")
        struct_errors = validate_stage1_output(
            round_name=round_name,
            generated=generated,
            shot_timestamps=shot_timestamps,
            references=references,
            events=events,
            video_duration=video_duration,
        )

        # Record structural validation results
        if recorder:
            recorder.save(round_name, round_idx,
                          f"structural_validation_attempt{attempt+1}.json",
                          {"errors": struct_errors, "error_count": len(struct_errors)})

        # If CRITICAL structural errors exist, skip Stage2 and directly retry
        critical_errors = [e for e in struct_errors if e.startswith("CRITICAL")]
        if critical_errors:
            logger.warning(
                f"[Round:{round_name}] {len(critical_errors)} CRITICAL structural errors, "
                f"skipping Stage2 and retrying Stage1"
            )
            print(f"  \u274c {len(critical_errors)} CRITICAL structural errors found, retrying...")
            for ce in critical_errors:
                print(f"      - {ce}")
            feedback = format_validation_feedback(struct_errors)
            continue

        # Non-critical structural errors: include as additional context for Stage2
        struct_feedback = ""
        if struct_errors:
            logger.info(
                f"[Round:{round_name}] {len(struct_errors)} non-critical structural issues"
            )
            print(f"  \u26a0\ufe0f  {len(struct_errors)} non-critical structural issues (proceeding to Stage2)")
            struct_feedback = format_validation_feedback(struct_errors)

        # --- Stage2: Evaluation ---
        print(f"  \U0001f50d Stage2 Eval ({len(models['stage2'])} evaluation models voting)...")
        logger.info(f"[Round:{round_name}] Stage2 evaluation ({len(models['stage2'])} evaluators)")
        verdict, eval_feedback, eval_details = run_stage2(
            models=models["stage2"],
            round_name=round_name,
            video_path=video_path,
            audio_path=audio_path,
            generated_content=generated,
            references=references,
            events=events,
            shot_timestamps=shot_timestamps,
            media_cache=media_cache,
        )

        # Record Stage2 evaluation results (including full details from each model)
        if recorder:
            recorder.save(round_name, round_idx,
                          f"stage2_eval_attempt{attempt+1}.json",
                          {"verdict": verdict, "feedback": eval_feedback,
                           "eval_details": eval_details})

        if verdict == "pass":
            # All pass -> proceed to Stage3
            print(f"  \u2705 Stage2 passed! All evaluation models agree.")
            logger.info(f"[Round:{round_name}] Stage2 PASSED")
            verified_content = generated
            # Record Stage2-verified content
            if recorder:
                recorder.save(round_name, round_idx,
                              "stage2_verified.json", verified_content)
            break
        elif verdict == "retry":
            # Partial failure -> fall back to Stage1 with feedback
            print(f"  \U0001f501 Stage2: Some evaluators found issues, retrying with feedback...")
            logger.info(f"[Round:{round_name}] Stage2 RETRY")
            # Combine eval feedback with structural feedback
            combined_fb = eval_feedback
            if struct_feedback:
                combined_fb = f"{eval_feedback}\n\n{struct_feedback}"
            feedback = combined_fb
            continue
        else:
            # discard: All fail -> discard this result
            print(f"  \u274c Stage2: All evaluation models found issues, discarding result.")
            logger.warning(f"[Round:{round_name}] Stage2 DISCARD (all evaluators failed)")
            summary_logger.warning(f"[Stage2-DISCARD] round={round_name}: all evaluators failed")
            # On discard, keep feedback but add stronger guidance
            # Extract critical and major issues as focus points
            discard_guidance = (
                "IMPORTANT: Your previous output was REJECTED by multiple evaluators. "
                "This means there are SERIOUS factual errors. "
                "Please generate a COMPLETELY NEW version, paying special attention to:\n"
                "1. Do NOT repeat the same errors listed below.\n"
                "2. Only describe what you can DIRECTLY OBSERVE in the video/audio.\n"
                "3. If you are unsure about any content, OMIT it rather than guessing.\n"
                "4. It is better to have fewer but correct items.\n\n"
            )
            combined_fb = discard_guidance + (eval_feedback if eval_feedback else "")
            if struct_feedback:
                combined_fb = f"{combined_fb}\n\n{struct_feedback}"
            feedback = combined_fb
            continue
    if verified_content is None:
        logger.error(f"[Round:{round_name}] All Stage1-Stage2 attempts exhausted")
        summary_logger.error(f"[EXHAUSTED] round={round_name}: all Stage1-Stage2 attempts failed")
        print(f"  \u274c Round {round_name} failed: could not pass Stage2 evaluation")
        raise PipelineRejectError(f"Round {round_name}: all Stage1-Stage2 attempts exhausted")

    # ================================================================
    # Stage3: Refine + re-evaluation loop
    # ================================================================
    print(f"  \u2728 Stage3 Refine (2 models refine independently)...")
    logger.info(f"[Round:{round_name}] Stage3 refinement")

    # Initial refinement: 2 models refine separately
    caption_a, caption_b = run_stage3(
        refine_models=models["stage3"],
        round_name=round_name,
        video_path=video_path,
        audio_path=audio_path,
        verified_content=verified_content,
        references=references,
        events=events,
        media_cache=media_cache,
    )

    # Record Stage3 initial refine results
    if recorder:
        recorder.save(round_name, round_idx,
                      "stage3_refine_a_initial.json", caption_a)
        recorder.save(round_name, round_idx,
                      "stage3_refine_b_initial.json", caption_b)

    # Re-evaluate both refined captions through Stage2
    final_a, final_b = caption_a, caption_b
    a_passed = False
    b_passed = False

    for refine_attempt in range(config.STAGE3_MAX_RETRY + 1):
        logger.info(
            f"[Round:{round_name}] Stage3 re-eval attempt {refine_attempt + 1}"
        )
        print(
            f"  \U0001f50d Stage3 re-eval (attempt {refine_attempt + 1}"
            f"/{config.STAGE3_MAX_RETRY + 1})..."
        )

        # Evaluate caption A and caption B in parallel (if both haven't passed yet)
        from concurrent.futures import ThreadPoolExecutor as _S3Pool, as_completed as _s3_done

        def _eval_caption(label, caption):
            """Evaluate a single caption through Stage2."""
            v, fb, ed = run_stage2(
                models=models["stage2"],
                round_name=round_name,
                video_path=video_path,
                audio_path=audio_path,
                generated_content=caption,
                references=references,
                events=events,
                shot_timestamps=shot_timestamps,
                media_cache=media_cache,
            )
            return label, v, fb, ed

        eval_tasks = []
        if not a_passed:
            eval_tasks.append(("A", final_a))
        if not b_passed:
            eval_tasks.append(("B", final_b))

        a_verdict, a_feedback, a_eval_details = None, "", []
        b_verdict, b_feedback, b_eval_details = None, "", []

        if eval_tasks:
            with _S3Pool(max_workers=len(eval_tasks)) as pool:
                futures = [pool.submit(_eval_caption, label, cap) for label, cap in eval_tasks]
                for future in _s3_done(futures):
                    label, verdict, feedback_text, eval_det = future.result()
                    if label == "A":
                        a_verdict = verdict
                        a_feedback = feedback_text
                        a_eval_details = eval_det
                        a_passed = (verdict == "pass")
                        logger.info(f"[Round:{round_name}] Refined A: {'PASS' if a_passed else 'FAIL'} (verdict={verdict})")
                        print("    Caption A: " + ("\u2705 Passed" if a_passed else "\u274c Failed"))
                        if recorder:
                            recorder.save(round_name, round_idx,
                                          f"stage3_reeval_a_attempt{refine_attempt+1}.json",
                                          {"passed": a_passed, "verdict": verdict,
                                           "feedback": feedback_text,
                                           "eval_details": eval_det,
                                           "caption_evaluated": final_a})
                    elif label == "B":
                        b_verdict = verdict
                        b_feedback = feedback_text
                        b_eval_details = eval_det
                        b_passed = (verdict == "pass")
                        logger.info(f"[Round:{round_name}] Refined B: {'PASS' if b_passed else 'FAIL'} (verdict={verdict})")
                        print("    Caption B: " + ("\u2705 Passed" if b_passed else "\u274c Failed"))
                        if recorder:
                            recorder.save(round_name, round_idx,
                                          f"stage3_reeval_b_attempt{refine_attempt+1}.json",
                                          {"passed": b_passed, "verdict": verdict,
                                           "feedback": feedback_text,
                                           "eval_details": eval_det,
                                           "caption_evaluated": final_b})

        # Both pass -> proceed to Stage4
        if a_passed and b_passed:
            print(f"  \u2705 Both refined captions passed evaluation!")
            logger.info(f"[Round:{round_name}] Both refined captions PASSED")
            break

        # Still have retry attempts -> fall back to Stage1 and regenerate
        if refine_attempt < config.STAGE3_MAX_RETRY:
            print(f"  \U0001f501 Falling back to Stage1, using refined captions as context to regenerate...")
            logger.info(f"[Round:{round_name}] Stage3 retry {refine_attempt+1}: re-running Stage1")

            # Use failed refined captions as feedback context
            refine_feedback_parts = []
            refine_feedback_parts.append(
                "The refined captions failed Stage2 evaluation. "
                "Please generate a new version that avoids the issues found. "
                "Focus on factual accuracy — only describe what you can directly observe."
            )
            if not a_passed and a_feedback:
                refine_feedback_parts.append(
                    f"\n--- Issues found in Refined Caption A ---\n"
                    f"{a_feedback}"
                )
            if not b_passed and b_feedback:
                refine_feedback_parts.append(
                    f"\n--- Issues found in Refined Caption B ---\n"
                    f"{b_feedback}"
                )
            refine_feedback = "\n".join(refine_feedback_parts)

            # Re-run Stage1 generation
            regenerated = run_stage1(
                model=models["stage1"],
                round_name=round_name,
                video_path=video_path,
                audio_path=audio_path,
                shot_timestamps=shot_timestamps,
                references=references,
                events=events,
                feedback=refine_feedback,
                media_cache=media_cache,
            )

            # Record Stage1 regeneration during Stage3 iteration
            if recorder:
                recorder.save(round_name, round_idx,
                              f"stage3_retry{refine_attempt+1}_stage1_regen.json",
                              {"feedback": refine_feedback, "regenerated": regenerated})

            if regenerated.get("parse_error"):
                logger.warning(f"[Round:{round_name}] Stage3 retry: Stage1 parse error")
                # Record parse failure
                if recorder:
                    recorder.save(round_name, round_idx,
                                  f"stage3_retry{refine_attempt+1}_stage1_parse_error.json",
                                  {"error": "parse_error", "regenerated": regenerated})
                continue

            # Re-evaluate through Stage2
            v2, fb2, v2_eval_details = run_stage2(
                models=models["stage2"],
                round_name=round_name,
                video_path=video_path,
                audio_path=audio_path,
                generated_content=regenerated,
                references=references,
                events=events,
                shot_timestamps=shot_timestamps,
                media_cache=media_cache,
            )

            # Record Stage2 re-evaluation during Stage3 iteration
            if recorder:
                recorder.save(round_name, round_idx,
                              f"stage3_retry{refine_attempt+1}_stage2_reeval.json",
                              {"verdict": v2, "feedback": fb2,
                               "eval_details": v2_eval_details,
                               "content_evaluated": regenerated})

            if v2 == "pass":
                verified_content = regenerated
                # Record updated verified_content
                if recorder:
                    recorder.save(round_name, round_idx,
                                  f"stage3_retry{refine_attempt+1}_new_verified.json",
                                  verified_content)

                # Re-refine (verified_content changed, both captions need regeneration and re-evaluation)
                new_a, new_b = run_stage3(
                    refine_models=models["stage3"],
                    round_name=round_name,
                    video_path=video_path,
                    audio_path=audio_path,
                    verified_content=verified_content,
                    references=references,
                    events=events,
                    media_cache=media_cache,
                )

                # Record re-refinement results
                if recorder:
                    recorder.save(round_name, round_idx,
                                  f"stage3_retry{refine_attempt+1}_refine_a.json", new_a)
                    recorder.save(round_name, round_idx,
                                  f"stage3_retry{refine_attempt+1}_refine_b.json", new_b)

                # verified_content changed, both captions are based on new content, both need re-evaluation
                final_a = new_a
                final_b = new_b
                a_passed = False
                b_passed = False
            else:
                logger.warning(
                    f"[Round:{round_name}] Stage3 retry: Stage2 did not pass "
                    f"(verdict={v2}), continuing iteration"
                )

    # ================================================================
    # Stage3 Fallback handling
    # ================================================================
    stage3_fallback = "both_passed"
    if a_passed and b_passed:
        # Both passed, proceed normally to Stage4
        pass
    elif a_passed and not b_passed:
        # Only A passed -> use A as both candidates
        print(f"  \u26a0\ufe0f  Only caption A passed, using A for Stage4")
        logger.info(f"[Round:{round_name}] Stage3 fallback: only A passed")
        stage3_fallback = "only_a_passed"
        final_b = final_a
    elif b_passed and not a_passed:
        # Only B passed -> use B as both candidates
        print(f"  \u26a0\ufe0f  Only caption B passed, using B for Stage4")
        logger.info(f"[Round:{round_name}] Stage3 fallback: only B passed")
        stage3_fallback = "only_b_passed"
        final_a = final_b
    else:
        # Neither passed -> fall back to the unrefined Stage2-verified version
        print(f"  \u26a0\ufe0f  Both refined captions failed, using unrefined Stage2 result")
        logger.warning(
            f"[Round:{round_name}] Stage3 fallback: using unrefined content"
        )
        stage3_fallback = "both_failed_use_unrefined"
        final_a = verified_content
        final_b = verified_content

    # Record Stage3 final status (including fallback info)
    if recorder:
        recorder.save(round_name, round_idx,
                      "stage3_final_status.json",
                      {"fallback": stage3_fallback,
                       "a_passed": a_passed, "b_passed": b_passed,
                       "total_refine_attempts": refine_attempt + 1,
                       "final_a": final_a, "final_b": final_b})

    # ================================================================
    # Stage4: Select (atomic unit selection)
    # Candidate sources:
    #   Candidate A = verified_content (original Stage2-verified version before refine)
    #   Candidate B = final_a (refined version 1)
    #   Candidate C = final_b (refined version 2, may be same as B)
    # ================================================================
    print(f"  \U0001f3af Stage4 Select (atomic unit selection)...")
    logger.info(f"[Round:{round_name}] Stage4 selection")

    # Determine the actual candidates for Stage4
    # A = original verified (always available)
    # B = refined version 1 (final_a)
    # C = refined version 2 (final_b), only if different from B
    stage4_a = verified_content  # Original verified (pre-refine)
    stage4_b = final_a           # Refined version 1
    stage4_c = final_b           # Refined version 2

    # Record candidates entering Stage4
    if recorder:
        recorder.save(round_name, round_idx,
                      "stage4_candidate_a_original.json", stage4_a)
        recorder.save(round_name, round_idx,
                      "stage4_candidate_b_refined1.json", stage4_b)
        recorder.save(round_name, round_idx,
                      "stage4_candidate_c_refined2.json", stage4_c)

    # Check how many distinct candidates we have
    a_json = json.dumps(stage4_a, sort_keys=True)
    b_json = json.dumps(stage4_b, sort_keys=True)
    c_json = json.dumps(stage4_c, sort_keys=True)

    all_same = (a_json == b_json == c_json)
    b_eq_c = (b_json == c_json)
    a_eq_b = (a_json == b_json)
    a_eq_c = (a_json == c_json)

    if all_same:
        # All three are identical -> skip selection
        print(f"  \u2139\ufe0f  All candidates are identical, skipping selection")
        logger.info(f"[Round:{round_name}] Stage4 skipped (all candidates identical)")
        round_result = stage4_a
        if recorder:
            recorder.save(round_name, round_idx,
                          "stage4_skipped.json",
                          {"reason": "all_candidates_identical",
                           "stage3_fallback": stage3_fallback})
    elif b_eq_c:
        # B and C are the same but different from A -> 2-way selection (A vs B)
        print(f"  \u2139\ufe0f  Both refined versions are identical, 2-way selection (original vs refined)...")
        logger.info(f"[Round:{round_name}] Stage4: 2-way (original vs refined, B==C)")
        round_result = run_stage4(
            model=models["stage4"],
            round_name=round_name,
            video_path=video_path,
            audio_path=audio_path,
            candidate_a=stage4_a,
            candidate_b=stage4_b,
            candidate_c=None,  # Only 2 candidates
            references=references,
            events=events,
            media_cache=media_cache,
        )
        if recorder:
            recorder.save(round_name, round_idx,
                          "stage4_select_result.json",
                          {"stage3_fallback": stage3_fallback,
                           "mode": "2-way (original vs refined)",
                           "selected_result": round_result})
    else:
        # All three are distinct -> 3-way selection
        actual_c = None if a_eq_c else stage4_c  # Don't pass C if it equals A
        if a_eq_b:
            # A == B but C is different -> 2-way (A vs C)
            num_way = 2
            print(f"  \u2139\ufe0f  Original and refined1 are identical, 2-way selection (original vs refined2)...")
            logger.info(f"[Round:{round_name}] Stage4: 2-way (A==B, selecting A vs C)")
            round_result = run_stage4(
                model=models["stage4"],
                round_name=round_name,
                video_path=video_path,
                audio_path=audio_path,
                candidate_a=stage4_a,
                candidate_b=stage4_c,
                candidate_c=None,
                references=references,
                events=events,
                media_cache=media_cache,
            )
        else:
            num_way = 3 if actual_c is not None else 2
            print(f"  \u2139\ufe0f  {num_way}-way selection (original + refined1 + refined2)...")
            logger.info(f"[Round:{round_name}] Stage4: {num_way}-way selection")
            round_result = run_stage4(
                model=models["stage4"],
                round_name=round_name,
                video_path=video_path,
                audio_path=audio_path,
                candidate_a=stage4_a,
                candidate_b=stage4_b,
                candidate_c=actual_c,
                references=references,
                events=events,
                media_cache=media_cache,
            )
        if recorder:
            recorder.save(round_name, round_idx,
                          "stage4_select_result.json",
                          {"stage3_fallback": stage3_fallback,
                           "mode": f"{num_way}-way",
                           "selected_result": round_result})

    # ---- Post-processing: fix sub_shot_time out-of-bounds (shot round only) ----
    if round_name == "shot" and round_result and "shots" in round_result:
        logger.info("[PostProcess] Running sub_shot_time bounds check and fix...")
        round_result = fix_sub_shot_times(round_result, shot_timestamps)
        if recorder:
            recorder.save(round_name, round_idx,
                          "postprocess_fix_sub_shot_times.json", round_result)

    # Record Stage4 final result
    if recorder:
        recorder.save(round_name, round_idx,
                      "stage4_final.json", round_result)

    elapsed = time.time() - round_start
    print(f"  \u2705 Round {round_name} complete! ({elapsed:.1f}s)")
    logger.info(
        f"[Round:{round_name}] Complete in {elapsed:.1f}s, "
        f"result keys={list(round_result.keys()) if round_result else 'None'}"
    )

    return round_result


# ======================================================================
# Full pipeline for a single video (3 rounds)
# ======================================================================

def run_pipeline_for_item(
    item: dict,
    item_index: int,
    total_items: int,
    models: dict = None,
) -> Optional[dict]:
    """
    Execute the full 3-round pipeline for a single video.

    Args:
        item:        Input data dictionary containing video_path, audio_path, etc.
        item_index:  Current item index (for logging)
        total_items: Total number of items (for progress display)
        models:      Pre-initialized model dictionary (None to create new)

    Returns:
        Complete MTSS dict (containing references, events, shots), or None on failure.
    """
    item_start = time.time()
    # DID: try multiple keys for compatibility with different data formats
    did = (item.get("DID")
           or item.get("video_id")
           or item.get("id")
           or f"item_{item_index}")

    # Video path: prefer timestamp_video_path (with red timestamp overlay),
    # fall back to video_path
    video_path = (item.get("timestamp_video_path", "")
                  or item.get("video_path", ""))
    audio_path = item.get("audio_path", "")

    print(f"\n{'='*70}")
    print(f"\U0001f4f9 Processing [{item_index + 1}/{total_items}]: {did}")
    print(f"{'='*70}")
    logger.info(f"Processing item {item_index + 1}/{total_items}: {did}")

    # Validate video file exists
    if not video_path or not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        print(f"  \u274c Video file not found: {video_path}")
        return None

    # Extract audio (if needed)
    if not audio_path or not os.path.exists(audio_path):
        print(f"  \U0001f50a Extracting audio from video...")
        audio_path = extract_audio_from_video(video_path)
        if audio_path:
            print(f"  \u2705 Audio extraction successful: {audio_path}")
        else:
            print(f"  \u26a0\ufe0f  No audio available")

    # Detect shot boundaries (use PySceneDetect auto-detection when shot_time is empty)
    print(f"  \U0001f3ac Detecting shot boundaries...")
    shot_timestamps = item.get("shot_time", None)
    if not shot_timestamps:
        logger.info(f"[{did}] shot_time not provided, using PySceneDetect auto-detection")
        print(f"  \U0001f50d shot_time not provided, using PySceneDetect auto-detection...")
        shot_timestamps = detect_shots(video_path)
        logger.info(f"[{did}] PySceneDetect detected {len(shot_timestamps)} shots: {shot_timestamps}")
    else:
        logger.info(f"[{did}] Using provided shot_time: {len(shot_timestamps)} shots")
    print(f"  \u2705 Shot boundaries: {len(shot_timestamps)} shots -> {shot_timestamps}")

    # Initialize models (if not provided)
    if models is None:
        models = _init_all_models()

    # Create media cache (same video is preprocessed only once across the entire pipeline)
    media_cache = MediaCache()

    # Set HTTP URL for Gemini/Qwen3.5-Omni-Plus models (from input data's url_t field)
    video_http_url = item.get("url_t", "") or item.get("video_url", "")
    if video_http_url:
        media_cache.set_video_http_url(video_http_url)

    # Create intermediate result recorder
    recorder = IntermediateRecorder(did)

    # Record input metadata (shot_timestamps, video_path, etc.)
    recorder.save("metadata", 0, "input_metadata.json", {
        "DID": did,
        "video_path": video_path,
        "audio_path": audio_path,
        "shot_timestamps": shot_timestamps,
        "shot_count": len(shot_timestamps),
    })

    # ---- Round-level checkpoint for resume ----
    # Check if we have saved round results from a previous run
    checkpoint_path = os.path.join(
        recorder.base_path, "round_checkpoint.json"
    ) if recorder.enabled else None

    saved_references = None
    saved_events = None
    resume_from_round = 1  # 1=reference_bank, 2=event, 3=shot

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            if ckpt.get("reference_bank_done"):
                saved_references = ckpt.get("references")
                resume_from_round = 2
                logger.info(f"[{did}] Checkpoint: reference_bank done, resuming from event")
                print(f"  \u267b\ufe0f  Checkpoint resume: reference_bank complete, starting from event round")
            if ckpt.get("event_done"):
                saved_events = ckpt.get("events")
                resume_from_round = 3
                logger.info(f"[{did}] Checkpoint: event done, resuming from shot")
                print(f"  \u267b\ufe0f  Checkpoint resume: event complete, starting from shot round")
        except Exception as e:
            logger.warning(f"[{did}] Failed to load checkpoint: {e}")

    def _save_round_checkpoint(ref=None, evt=None, ref_done=False, evt_done=False):
        """Save round-level checkpoint for resume."""
        if not checkpoint_path:
            return
        try:
            ckpt_data = {
                "reference_bank_done": ref_done,
                "event_done": evt_done,
                "references": ref,
                "events": evt,
            }
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(ckpt_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"[{did}] Round checkpoint saved")
        except Exception as e:
            logger.warning(f"[{did}] Failed to save round checkpoint: {e}")

    # ---- Round 1: Reference Bank ----
    if resume_from_round <= 1:
        print("\n" + "\u2500" * 50)
        print(f"  \U0001f3f7\ufe0f  Round 1/3: Reference Bank")
        print("\u2500" * 50)
        ref_result = _run_single_round(
            models=models,
            round_name="reference_bank",
            video_path=video_path,
            audio_path=audio_path,
            shot_timestamps=shot_timestamps,
            media_cache=media_cache,
            recorder=recorder,
            round_idx=1,
        )
        if ref_result is None:
            print(f"  \u274c Pipeline failed at reference_bank round")
            raise PipelineRejectError("reference_bank round returned None")

        references = ref_result.get("references", ref_result)
        # Save checkpoint after reference_bank
        _save_round_checkpoint(ref=references, ref_done=True)
    else:
        references = saved_references
        print(f"  \u2705 Round 1/3: Reference Bank (restored from checkpoint)")

    # ---- Round 2: Event ----
    if resume_from_round <= 2:
        print("\n" + "\u2500" * 50)
        print(f"  \U0001f3b5 Round 2/3: Event")
        print("\u2500" * 50)
        event_result = _run_single_round(
            models=models,
            round_name="event",
            video_path=video_path,
            audio_path=audio_path,
            shot_timestamps=shot_timestamps,
            media_cache=media_cache,
            references=references,
            recorder=recorder,
            round_idx=2,
        )
        if event_result is None:
            print(f"  \u274c Pipeline failed at event round")
            raise PipelineRejectError("event round returned None")

        events = event_result.get("events", event_result)
        # Save checkpoint after event
        _save_round_checkpoint(ref=references, evt=events, ref_done=True, evt_done=True)
    else:
        events = saved_events
        print(f"  \u2705 Round 2/3: Event (restored from checkpoint)")

    # ---- Round 3: Shot (with optional chunking for long videos) ----
    print("\n" + "\u2500" * 50)
    print(f"  \U0001f3ac Round 3/3: Shot")
    print("\u2500" * 50)

    chunk_size = getattr(config, 'SHOT_CHUNK_SIZE', 8)
    chunk_threshold = getattr(config, 'SHOT_CHUNK_THRESHOLD', 10)
    num_shots = len(shot_timestamps)

    if chunk_size > 0 and chunk_threshold > 0 and num_shots > chunk_threshold:
        # ---- Chunked shot processing for long videos ----
        # Split shot_timestamps into groups, each group runs full Stage1-4 independently
        chunks = []
        for start_idx in range(0, num_shots, chunk_size):
            end_idx = min(start_idx + chunk_size, num_shots)
            chunks.append((start_idx, end_idx, shot_timestamps[start_idx:end_idx]))

        print(f"  \U0001f4e6 Long video chunked processing: {num_shots} shots -> {len(chunks)} groups (max {chunk_size} per group)")
        logger.info(
            f"[Shot-Chunking] {num_shots} shots > threshold {chunk_threshold}, "
            f"splitting into {len(chunks)} chunks of max {chunk_size}"
        )

        all_shots = []
        chunk_failed = False

        for chunk_idx, (start_idx, end_idx, chunk_timestamps) in enumerate(chunks):
            chunk_label = f"chunk_{chunk_idx+1}/{len(chunks)}"
            print("\n  " + "\u00b7" * 40)
            print(f"  \U0001f4e6 Shot {chunk_label}: shots {start_idx+1}-{end_idx} "
                  f"({len(chunk_timestamps)} shots)")
            print("  " + "\u00b7" * 40)
            logger.info(
                f"[Shot-Chunking] Processing {chunk_label}: "
                f"shots {start_idx+1}-{end_idx}, timestamps={chunk_timestamps}"
            )

            # Pass FULL references and events to every chunk to ensure
            # consistent ref_id / event_id references across all chunks.
            # Only shot_timestamps differs per chunk — the model will generate
            # shots only for the given time segments and correctly reference
            # the full events/references.

            try:
                chunk_result = _run_single_round(
                    models=models,
                    round_name="shot",
                    video_path=video_path,
                    audio_path=audio_path,
                    shot_timestamps=chunk_timestamps,
                    media_cache=media_cache,
                    references=references,
                    events=events,  # FULL events, not filtered
                    recorder=recorder,
                    round_idx=3,
                )
            except PipelineRejectError as e:
                logger.error(f"[Shot-Chunking] {chunk_label} rejected: {e}")
                print(f"  \u274c Shot {chunk_label} validation failed")
                chunk_failed = True
                break

            if chunk_result is None:
                logger.error(f"[Shot-Chunking] {chunk_label} returned None")
                print(f"  \u274c Shot {chunk_label} failed")
                chunk_failed = True
                break

            chunk_shots = chunk_result.get("shots", [])
            if not isinstance(chunk_shots, list):
                chunk_shots = []

            # Re-number shot_ids to be globally sequential
            for j, shot in enumerate(chunk_shots):
                if isinstance(shot, dict):
                    shot["shot_id"] = f"shot_{start_idx + j + 1}"
                    # Re-map active_events back to original event_ids
                    # (chunk_events may have been filtered, but event_ids are unchanged)

            all_shots.extend(chunk_shots)
            print(f"  \u2705 Shot {chunk_label} complete: {len(chunk_shots)} shots")
            logger.info(
                f"[Shot-Chunking] {chunk_label} done: {len(chunk_shots)} shots"
            )

            # Save chunk intermediate result
            if recorder:
                recorder.save("shot", 3,
                              f"shot_chunk_{chunk_idx+1}_result.json",
                              {"chunk_idx": chunk_idx,
                               "start_idx": start_idx, "end_idx": end_idx,
                               "chunk_timestamps": chunk_timestamps,
                               "chunk_shots": chunk_shots})

        if chunk_failed:
            print(f"  \u274c Pipeline failed during shot round chunked processing")
            raise PipelineRejectError("shot round chunked processing failed")

        shots = all_shots
        print(f"\n  \u2705 Shot chunked processing complete: {len(shots)} total shots")
        logger.info(f"[Shot-Chunking] All chunks done: {len(shots)} total shots")

        # Save merged result
        if recorder:
            recorder.save("shot", 3, "shot_chunks_merged.json",
                          {"total_chunks": len(chunks),
                           "total_shots": len(shots),
                           "shots": shots})
    else:
        # ---- Normal (non-chunked) shot processing ----
        shot_result = _run_single_round(
            models=models,
            round_name="shot",
            video_path=video_path,
            audio_path=audio_path,
            shot_timestamps=shot_timestamps,
            media_cache=media_cache,
            references=references,
            events=events,
            recorder=recorder,
            round_idx=3,
        )
        if shot_result is None:
            print(f"  \u274c Pipeline failed at shot round")
            raise PipelineRejectError("shot round returned None")

        shots = shot_result.get("shots", shot_result)

    # ---- Assemble final MTSS ----
    mtss = {
        "references": references if isinstance(references, list) else references,
        "events": events if isinstance(events, list) else events,
        "shots": shots if isinstance(shots, list) else shots,
        "shot_time": shot_timestamps,  # Record the shot boundaries used
    }

    # Record final result
    recorder.save_final(mtss)

    # Clear media cache
    media_cache.clear()

    elapsed = time.time() - item_start
    print(f"\n\U0001f389 Pipeline complete: {did}! ({elapsed:.1f}s)")
    logger.info(f"Pipeline complete for {did} in {elapsed:.1f}s")

    return mtss


# ======================================================================
# Stratified sampling utility
# ======================================================================

def _stratified_sample(data: list, n: int, bucket_key: str = "duration_bucket") -> tuple:
    """
    Stratified sampling: evenly sample N items across all duration_bucket categories.
    Each bucket gets floor(N / num_buckets) items, remainder distributed round-robin.
    Within each bucket, items are randomly sampled.

    Args:
        data:       Full dataset list
        n:          Total number of items to sample
        bucket_key: Key in each item dict for stratification (default: "duration_bucket")

    Returns:
        Tuple of (sampled_items, original_indices) where original_indices are the
        indices in the original data list. This allows stable DID generation and
        resume across runs.
    """
    import random
    from collections import defaultdict

    # Group items by bucket, preserving original indices
    buckets = defaultdict(list)
    for orig_idx, item in enumerate(data):
        bk = item.get(bucket_key, "unknown")
        buckets[bk].append((orig_idx, item))

    bucket_names = sorted(buckets.keys())
    num_buckets = len(bucket_names)

    if num_buckets == 0:
        indices = list(range(min(n, len(data))))
        return [data[i] for i in indices], indices

    # Calculate per-bucket quota
    base_quota = n // num_buckets
    remainder = n % num_buckets

    sampled = []
    for i, bk in enumerate(bucket_names):
        pool = buckets[bk]
        quota = base_quota + (1 if i < remainder else 0)
        quota = min(quota, len(pool))  # Don't exceed available items
        random.shuffle(pool)
        chosen = pool[:quota]
        sampled.extend(chosen)
        logger.info(f"[Stratified Sample] bucket='{bk}': {len(pool)} available, sampled {len(chosen)}")
        print(f"  \U0001f4e6 {bk}: {len(chosen)}/{len(pool)} items")

    # Shuffle the final sample to avoid bucket-ordered processing
    random.shuffle(sampled)
    logger.info(f"[Stratified Sample] Total sampled: {len(sampled)} from {len(data)} items across {num_buckets} buckets")

    # Separate indices and items
    original_indices = [idx for idx, _ in sampled]
    sampled_items = [item for _, item in sampled]
    return sampled_items, original_indices


# ======================================================================
# Batch pipeline runner
# ======================================================================

def run_pipeline(
    input_json_path: str = None,
    output_dir: str = None,
    resume: bool = True,
    num_workers: int = None,
    limit: int = None,
    sample: int = None,
    skip_failed: bool = False,
) -> None:
    """
    Execute the full pipeline on a batch of video items, with multi-threaded parallel processing.

    Args:
        input_json_path: Path to the input JSON file (list of items)
        output_dir:      Output directory for results
        resume:          Whether to skip already-processed items (checkpoint resume)
        num_workers:     Number of parallel worker threads (defaults to config.NUM_WORKERS)
        limit:           Only process the first N items (None means process all)
        sample:          Stratified sample N items by duration_bucket (takes priority over limit)
        skip_failed:     Whether to skip failed items on resume (default False, failed items are retried)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    import fcntl

    input_path = input_json_path or config.DEFAULT_INPUT_JSON
    out_dir = output_dir or config.DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    workers = num_workers or getattr(config, "NUM_WORKERS", 1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_file = os.path.join(out_dir, "progress.json")

    # Determine output file: resume from previous output file if available
    output_file = None
    if resume and os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                prev_progress = json.load(f)
            prev_output = prev_progress.get("output_file", "")
            if prev_output and os.path.exists(prev_output):
                output_file = prev_output
                logger.info(f"[Resume] Continuing to write to previous output file: {output_file}")
                print(f"\U0001f4c2 Checkpoint resume: continuing to write to previous output file")
        except Exception:
            pass

    if output_file is None:
        output_file = os.path.join(out_dir, f"mtss_results_{timestamp}.jsonl")

    print(f"{'='*70}")
    print(f"\U0001f680 MTSS Pipeline - Batch Processing")
    print(f"{'='*70}")
    print(f"  Input:     {input_path}")
    print(f"  Output:    {output_file}")
    print(f"  Workers:   {workers}")
    print(f"  Resume:    {resume}")
    if resume and skip_failed:
        print(f"  Skip failed: Yes (only process unhandled items)")
    elif resume:
        print(f"  Skip failed: No (failed items will be retried)")
    print(f"{'='*70}")

    # Load input data
    with open(input_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    if not isinstance(all_data, list):
        print("\u274c Input JSON must be a list")
        return

    # ---- Load progress (resume) ----
    # Must load progress BEFORE sampling, so we can restore the previous sample
    success_dids = set()
    rejected_dids = set()  # validation rejected - can be skipped with --skip-failed
    error_dids = set()     # API/network error - always retry
    saved_sampled_indices = None  # Original data indices from previous run's sampling

    if resume and os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                progress = json.load(f)
            if "success_dids" in progress:
                success_dids = set(progress.get("success_dids", []))
                rejected_dids = set(progress.get("rejected_dids", []))
                error_dids = set(progress.get("error_dids", []))
                # Backward compatible: migrate old failed_dids to error_dids (assume they need retry)
                old_failed = set(progress.get("failed_dids", []))
                if old_failed and not rejected_dids and not error_dids:
                    error_dids = old_failed
                    logger.info(f"Migrated {len(old_failed)} old failed_dids to error_dids")
                # Restore previous sampling indices (critical for resume consistency)
                saved_sampled_indices = progress.get("sampled_indices", None)
                if skip_failed:
                    _resume_msg = f"[Resume] success {len(success_dids)}, rejected {len(rejected_dids)} (skip), API error {len(error_dids)} (retry)"
                    print(f"\U0001f4c2 Checkpoint resume: success {len(success_dids)}, rejected {len(rejected_dids)} (skip), API error {len(error_dids)} (retry)")
                else:
                    _resume_msg = f"[Resume] success {len(success_dids)}, rejected {len(rejected_dids)} (retry), API error {len(error_dids)} (retry)"
                    print(f"\U0001f4c2 Checkpoint resume: success {len(success_dids)}, rejected {len(rejected_dids)} (retry), API error {len(error_dids)} (retry)")
                summary_logger.info(_resume_msg)
            else:
                # Old format: treat all processed_dids as success (backward compatible)
                success_dids = set(progress.get("processed_dids", []))
                print(f"\U0001f4c2 Checkpoint resume: {len(success_dids)} processed (old format, all treated as success)")
                summary_logger.info(f"[Resume] processed {len(success_dids)} (old format)")
        except Exception:
            pass

    # ---- Apply stratified sampling or limit ----
    # Key: use original data indices as DID to ensure consistency across runs.
    # sampled_indices maps each position in `data` to its original index in `all_data`.
    total_before_sample = len(all_data)
    sampled_indices = None  # Will be set if sampling is used

    if resume and saved_sampled_indices is not None:
        # Resume mode: restore the exact same sample from previous run
        # This ensures DID consistency (item_88 always refers to the same video)
        valid_indices = [idx for idx in saved_sampled_indices if idx < len(all_data)]
        data = [all_data[idx] for idx in valid_indices]
        sampled_indices = valid_indices
        print(f"\U0001f4c2 Checkpoint resume: restored {len(data)} sampled items (original indices)")
        logger.info(f"[Resume] Restored {len(data)} sampled items from previous run")
    elif sample is not None and sample > 0:
        data, sampled_indices = _stratified_sample(all_data, sample)
        print(f"\U0001f3b2 Stratified sampling: sampled {len(data)} items from {total_before_sample}")
    elif limit is not None and limit > 0:
        data = all_data[:limit]
        sampled_indices = list(range(len(data)))
        print(f"\U0001f4ca Total items: {len(data)} (limited to first {limit})")
    else:
        data = all_data
        sampled_indices = list(range(len(data)))
        print(f"\U0001f4ca Total items: {len(data)}")
    total = len(data)

    # ---- Filter pending items ----
    # DID uses original data index (from sampled_indices) to ensure cross-run consistency.
    # e.g., item_88 always refers to all_data[88], regardless of sampling order.
    pending_items = []
    skipped_dids = success_dids.copy()
    if skip_failed:
        skipped_dids = skipped_dids | rejected_dids
    for i, item in enumerate(data):
        # Use original data index as DID for consistency
        orig_idx = sampled_indices[i] if sampled_indices and i < len(sampled_indices) else i
        did = (item.get("DID")
               or item.get("video_id")
               or item.get("id")
               or f"item_{orig_idx}")
        if resume and did in skipped_dids:
            continue
        pending_items.append((i, item))

    if not pending_items:
        print("\u2705 All items have been processed")
        summary_logger.info("[Progress] All items processed, nothing pending")
        return

    print(f"\U0001f4cb Pending: {len(pending_items)} items")
    _skipped_count = total - len(pending_items)
    summary_logger.info(
        f"[Progress] Total {total} | Pending {len(pending_items)} | "
        f"Skipped {_skipped_count} (success {len(success_dids)}, "
        f"rejected {len(rejected_dids) if skip_failed else 0})"
    )

    # Initialize all models (only once, shared across threads in a thread-safe manner)
    print(f"\U0001f527 Initializing models...")
    models = _init_all_models()
    print(f"\u2705 Models ready")

    # Thread-safe counters and write lock
    _lock = threading.Lock()
    _success_count = [0]
    _fail_count = [0]
    _failed_items = []
    _pipeline_start_time = time.time()
    _last_progress_report = [time.time()]  # mutable for closure access

    def _write_result_line(result_item: dict):
        """Append one result line to the JSONL file in real-time (thread-safe)."""
        with _lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

    def _update_progress(did: str, status: str):
        """Update progress file (thread-safe).
        
        Args:
            did: Data item ID
            status: One of 'success', 'rejected' (validation failed), 'error' (API/network error)
        """
        with _lock:
            # Remove from all sets first
            success_dids.discard(did)
            rejected_dids.discard(did)
            error_dids.discard(did)
            # Add to the correct set
            if status == "success":
                success_dids.add(did)
            elif status == "rejected":
                rejected_dids.add(did)
            else:  # error
                error_dids.add(did)
            try:
                progress = {
                    "success_dids": list(success_dids),
                    "rejected_dids": list(rejected_dids),
                    "error_dids": list(error_dids),
                    "sampled_indices": sampled_indices,  # Preserve sampling for resume
                    "last_update": datetime.now().isoformat(),
                    "success": _success_count[0],
                    "fail": _fail_count[0],
                    "total_pending": len(pending_items),
                    "output_file": output_file,  # Record output file path for resume
                }
                tmp = progress_file + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(progress, f, ensure_ascii=False, indent=2)
                os.replace(tmp, progress_file)
            except Exception as e:
                logger.warning(f"Failed to save progress: {e}")

    def _maybe_report_progress():
        """Periodic progress report: log a summary every 5 minutes (thread-safe)."""
        now = time.time()
        with _lock:
            if now - _last_progress_report[0] < 300:  # 5 minutes
                return
            _last_progress_report[0] = now
            elapsed = now - _pipeline_start_time
            done = _success_count[0] + _fail_count[0]
            remaining = len(pending_items) - done
            rate = done / elapsed * 3600 if elapsed > 0 else 0
            eta_hours = remaining / (done / elapsed) / 3600 if done > 0 and elapsed > 0 else float('inf')
            msg = (
                f"[Progress Report] "
                f"Done: {done}/{len(pending_items)} | "
                f"\u2705 {_success_count[0]} | \u274c {_fail_count[0]} | "
                f"Rate: {rate:.1f}/hr | "
                f"Elapsed: {elapsed/60:.1f}min | "
                f"ETA: {eta_hours:.1f}hr"
            )
            logger.info(msg)
            summary_logger.info(msg)
            print(f"\n\u23f1\ufe0f  {msg}")

    def _process_one(idx_item):
        """Process a single video (executed in the thread pool)."""
        i, item = idx_item
        # Use original data index as DID for cross-run consistency
        orig_idx = sampled_indices[i] if sampled_indices and i < len(sampled_indices) else i
        did = (item.get("DID")
               or item.get("video_id")
               or item.get("id")
               or f"item_{orig_idx}")
        try:
            mtss = run_pipeline_for_item(
                item=item,
                item_index=orig_idx,
                total_items=total,
                models=models,
            )
            if mtss is not None:
                result_item = {**item, "mtss_caption": mtss}
                _write_result_line(result_item)
                with _lock:
                    _success_count[0] += 1
                _update_progress(did, status="success")
                _maybe_report_progress()
                return did, True
            else:
                # run_pipeline_for_item returned None without raising
                # (e.g. video file not found) - treat as error (retryable)
                summary_logger.error(f"[ERROR] {did}: pipeline returned None (video not found or other issue)")
                with _lock:
                    _fail_count[0] += 1
                    _failed_items.append({"DID": did, "index": i, "reason": "pipeline_returned_none", "fail_type": "error"})
                _update_progress(did, status="error")
                _maybe_report_progress()
                return did, False
        except PipelineRejectError as e:
            # Validation rejected: all attempts exhausted, content quality insufficient
            # --skip-failed will skip these on next run
            logger.warning(f"[{did}] Validation rejected: {e}")
            summary_logger.warning(f"[REJECTED] {did}: {e}")
            print(f"  \u274c [{did}] Validation rejected: {e}")
            with _lock:
                _fail_count[0] += 1
                _failed_items.append({"DID": did, "index": i, "reason": str(e), "fail_type": "rejected"})
            _update_progress(did, status="rejected")
            _maybe_report_progress()
            return did, False
        except Exception as e:
            # API/network error or unexpected exception - should be retried
            logger.error(f"Exception while processing {did} (error): {e}", exc_info=True)
            summary_logger.error(f"[ERROR] {did}: {e}")
            print(f"  \u274c [{did}] Error (retryable): {e}")
            with _lock:
                _fail_count[0] += 1
                _failed_items.append({"DID": did, "index": i, "reason": str(e), "fail_type": "error"})
            _update_progress(did, status="error")
            _maybe_report_progress()
            return did, False

    # Execute processing
    if workers <= 1:
        # Serial mode (for debugging)
        for idx_item in pending_items:
            did, ok = _process_one(idx_item)
            done = _success_count[0] + _fail_count[0]
            print(f"\n\U0001f4ca Progress: {done}/{len(pending_items)} (\u2705 {_success_count[0]} / \u274c {_fail_count[0]})")
    else:
        # Parallel mode
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_one, idx_item): idx_item
                for idx_item in pending_items
            }
            for future in as_completed(futures):
                did, ok = future.result()
                done = _success_count[0] + _fail_count[0]
                print(f"\n\U0001f4ca Progress: {done}/{len(pending_items)} (\u2705 {_success_count[0]} / \u274c {_fail_count[0]})")

    # Save failed items list (for re-running later)
    if _failed_items:
        failed_file = os.path.join(out_dir, f"failed_items_{timestamp}.json")
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(_failed_items, f, ensure_ascii=False, indent=2)
        print(f"  Failed items log: {failed_file}")
        logger.info(f"Failed items saved to {failed_file}")

    print(f"\n{'='*70}")
    print(f"\U0001f3c1 Pipeline complete!")
    print(f"{'='*70}")
    # Count rejected vs error failures
    _rejected_count = sum(1 for fi in _failed_items if fi.get("fail_type") == "rejected")
    _error_count = sum(1 for fi in _failed_items if fi.get("fail_type") == "error")
    print(f"  Total:      {len(pending_items)}")
    print(f"  Success:    {_success_count[0]}")
    print(f"  Failed:     {_fail_count[0]}")
    print(f"    Rejected:   {_rejected_count} (--skip-failed to skip)")
    print(f"    API/Error:  {_error_count} (will be retried on resume)")
    print(f"  Output:     {output_file}")
    print(f"{'='*70}")

    _final_msg = (
        f"[FINAL] Pipeline complete: "
        f"total {len(pending_items)} | success {_success_count[0]} | "
        f"failed {_fail_count[0]} (rejected {_rejected_count}, error {_error_count}) | "
        f"output {output_file}"
    )
    logger.info(
        f"Pipeline complete: {_success_count[0]}/{len(pending_items)} success, "
        f"{_fail_count[0]} failed. Output: {output_file}"
    )
    summary_logger.info(_final_msg)
