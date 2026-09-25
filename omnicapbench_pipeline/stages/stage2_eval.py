#!/usr/bin/env python3
"""
MTSS Pipeline - Stage2: Factual Evaluation
Stage2 factual evaluation phase.
Uses 2-3 models to perform factual verification on Stage1 output.

Core principles:
  - Each evaluation model must wait until the API successfully returns a valid evaluation result
  - API errors (empty responses, HTTP 429/400, etc.) are automatically retried and do not count as evaluation failures
  - All models must pass (is_correct=True) for the result to be considered pass
  - Voting logic: all pass→pass, any pass→retry(with feedback), all fail→discard
"""
import json
import logging
import time as _time
import random
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .. import config
from ..model_client import (
    ModelClient, BaseModelClient, parse_response_to_json,
    create_model_client, INPUT_TOO_LONG_SENTINEL,
)
from ..media_utils import build_multimodal_content
from ..prompts.stage2_eval import get_eval_system_prompt, get_eval_user_prompt

logger = logging.getLogger("omnicapbench_pipeline")

# Max attempts for a single eval model to get a valid API response
_EVAL_API_MAX_ATTEMPTS = getattr(config, 'API_MAX_RETRIES', 100)
_EVAL_API_RETRY_DELAY = getattr(config, 'API_RETRY_DELAY', 10.0)


def _eval_single_with_retry(idx: int, model, content: list, sys_prompt: str) -> Tuple[int, dict]:
    """
    Call a single eval model and retry until we get a valid response.
    API errors (empty response, parse failure) trigger retries with exponential backoff.
    Only returns when we have a genuine evaluation result (is_correct = True or False).

    Special handling for deterministic errors:
      - Input too long (InvalidParameter/Range of input length): skip immediately
      - Content safety (DataInspectionFailed): retry up to 3 times then skip

    Returns:
        (idx, parsed_result) where parsed_result always has a valid is_correct field.
    """
    _consecutive_empty = 0  # Track consecutive empty responses to detect persistent issues
    _MAX_CONSECUTIVE_EMPTY = 10  # If we get 10 empty responses in a row, likely a persistent issue

    for attempt in range(_EVAL_API_MAX_ATTEMPTS):
        try:
            logger.info(
                f"[Stage2-Eval] Calling evaluation model {idx} ({model.tag}), "
                f"API attempt {attempt+1}/{_EVAL_API_MAX_ATTEMPTS}"
            )
            raw = model.inference(content, system_prompt=sys_prompt)

            # Check for INPUT_TOO_LONG sentinel — deterministic failure, no retry
            if raw == INPUT_TOO_LONG_SENTINEL:
                logger.warning(
                    f"[Stage2-Eval] Evaluation model {idx} ({model.tag}): "
                    f"input exceeds token limit (INPUT_TOO_LONG), returning immediately for fallback"
                )
                return idx, {
                    "is_correct": False,
                    "_input_too_long": True,
                    "_model_type": getattr(model, 'tag', 'unknown'),
                    "issues": [{"description": f"Input exceeds model token limit for {model.tag}"}],
                }

            if not raw:
                _consecutive_empty += 1
                logger.warning(
                    f"[Stage2-Eval] Evaluation model {idx} ({model.tag}): "
                    f"API returned empty response (attempt {attempt+1}, consecutive_empty={_consecutive_empty}), retrying..."
                )
                # If we get too many consecutive empty responses, it's likely a persistent issue
                # (e.g., input too long causing the model client to give up)
                if _consecutive_empty >= _MAX_CONSECUTIVE_EMPTY:
                    logger.error(
                        f"[Stage2-Eval] Evaluation model {idx} ({model.tag}): "
                        f"{_consecutive_empty} consecutive empty responses, likely a persistent issue "
                        f"(input too long or content safety). Giving up."
                    )
                    return idx, {
                        "is_correct": False,
                        "_api_exhausted": True,
                        "_persistent_empty": True,
                        "issues": [{"description": f"Persistent empty response after {_consecutive_empty} attempts (likely input too long or content safety)"}],
                    }
                if attempt < _EVAL_API_MAX_ATTEMPTS - 1:
                    delay = min(_EVAL_API_RETRY_DELAY * (1.5 ** attempt), 120.0)
                    jitter = random.uniform(0, delay * 0.3)
                    _time.sleep(delay + jitter)
                continue

            # Got a non-empty response, reset consecutive empty counter
            _consecutive_empty = 0

            parsed = parse_response_to_json(raw)

            if parsed.get("parse_error"):
                logger.warning(
                    f"[Stage2-Eval] Evaluation model {idx} ({model.tag}): "
                    f"JSON parse failed (attempt {attempt+1}), retrying..."
                )
                if attempt < _EVAL_API_MAX_ATTEMPTS - 1:
                    delay = min(_EVAL_API_RETRY_DELAY * (1.5 ** attempt), 120.0)
                    jitter = random.uniform(0, delay * 0.3)
                    _time.sleep(delay + jitter)
                continue

            # Valid response — we have a genuine evaluation
            is_ok = parsed.get("is_correct", False)
            logger.info(
                f"[Stage2-Eval] Evaluation model {idx} ({model.tag}): is_correct={is_ok}"
            )
            # Log raw response for debug
            logger.debug(
                f"[Stage2-Eval] Evaluation model {idx} raw response (length={len(raw)}):\n"
                f"{raw}"
            )
            return idx, parsed

        except Exception as e:
            logger.warning(
                f"[Stage2-Eval] Evaluation model {idx} ({model.tag}): "
                f"exception on attempt {attempt+1}: {e}"
            )
            if attempt < _EVAL_API_MAX_ATTEMPTS - 1:
                delay = min(_EVAL_API_RETRY_DELAY * (1.5 ** attempt), 120.0)
                jitter = random.uniform(0, delay * 0.3)
                _time.sleep(delay + jitter)
            continue

    # Exhausted all attempts — this should be very rare with 100 retries
    logger.error(
        f"[Stage2-Eval] Evaluation model {idx} ({model.tag}): "
        f"all {_EVAL_API_MAX_ATTEMPTS} API attempts failed"
    )
    return idx, {
        "is_correct": False,
        "_api_exhausted": True,
        "issues": [{"description": f"All {_EVAL_API_MAX_ATTEMPTS} API attempts failed"}],
    }


def _handle_input_too_long_fallback(
    eval_results: list,
    models: list,
    content: list,
    sys_prompt: str,
) -> tuple:
    """
    Smart fallback routing for INPUT_TOO_LONG errors.

    When a model (e.g., qwen3.5-omni-plus with 196608 token limit) fails because
    the input exceeds its token limit, this function automatically replaces it with
    fallback models that have larger context windows (e.g., gemini_25/gemini_31 with 1M+).

    The fallback model is selected from config.STAGE2_FALLBACK_MAP based on the
    failed model's model_type. To avoid duplicating an already-active evaluator,
    the function picks a fallback model_type that is NOT already in the current
    evaluator list.

    Args:
        eval_results: List of evaluation results from the initial run
        models:       List of model clients used in the initial run
        content:      The multimodal content (video + text prompt)
        sys_prompt:   The system prompt for evaluation

    Returns:
        (updated_eval_results, updated_models) — with fallback results replacing
        the INPUT_TOO_LONG entries.
    """
    fallback_map = getattr(config, 'STAGE2_FALLBACK_MAP', {})
    if not fallback_map:
        return eval_results, models

    # Find indices that hit INPUT_TOO_LONG
    too_long_indices = [
        i for i, r in enumerate(eval_results)
        if r and r.get("_input_too_long")
    ]

    if not too_long_indices:
        return eval_results, models

    logger.info(
        f"[Stage2-Eval] {len(too_long_indices)} model(s) hit INPUT_TOO_LONG, "
        f"initiating fallback routing..."
    )

    # Collect model_types already in use by non-failed evaluators
    active_model_types = set()
    for i, m in enumerate(models):
        if i not in too_long_indices:
            # Infer model_type from the client class
            mt = _get_model_type(m)
            if mt:
                active_model_types.add(mt)

    # For each failed model, find a suitable fallback
    fallback_tasks = []  # (idx, fallback_model_client)
    for idx in too_long_indices:
        failed_model = models[idx]
        failed_type = _get_model_type(failed_model)
        fallback_presets = fallback_map.get(failed_type, [])

        if not fallback_presets:
            logger.warning(
                f"[Stage2-Eval] No fallback configured for model_type='{failed_type}', "
                f"keeping original failure result for evaluator {idx}"
            )
            continue

        # Pick a fallback that is NOT already active
        chosen_preset = None
        for preset in fallback_presets:
            preset_type = preset.get("model_type", "")
            if preset_type not in active_model_types:
                chosen_preset = preset
                break

        if chosen_preset is None:
            # All fallbacks are already active; use the first one anyway
            # (better to have a duplicate evaluator than no evaluator)
            chosen_preset = fallback_presets[0]
            logger.info(
                f"[Stage2-Eval] All fallback model_types already active, "
                f"using {chosen_preset.get('model_type')} as duplicate for evaluator {idx}"
            )

        # Create the fallback model client
        fallback_tag = f"stage2_eval_{idx}_fallback"
        fallback_model = create_model_client(chosen_preset, tag=fallback_tag)
        fallback_tasks.append((idx, fallback_model))

        # Track the new model_type as active
        active_model_types.add(chosen_preset.get("model_type", ""))

        logger.info(
            f"[Stage2-Eval] Evaluator {idx}: replacing {failed_type} "
            f"with {chosen_preset.get('model_type')} (tag={fallback_tag})"
        )
        print(
            f"  🔄 Evaluator {idx}: {failed_type} hit token limit, "
            f"falling back to {chosen_preset.get('model_type')}"
        )

    if not fallback_tasks:
        return eval_results, models

    # Run fallback evaluations in parallel
    updated_results = list(eval_results)
    updated_models = list(models)

    with ThreadPoolExecutor(max_workers=len(fallback_tasks)) as executor:
        futures = [
            executor.submit(_eval_single_with_retry, idx, fb_model, content, sys_prompt)
            for idx, fb_model in fallback_tasks
        ]
        for future in as_completed(futures):
            idx, parsed = future.result()
            updated_results[idx] = parsed
            # Update the model reference so logging shows the correct model
            for task_idx, fb_model in fallback_tasks:
                if task_idx == idx:
                    updated_models[idx] = fb_model
                    break

    logger.info(
        f"[Stage2-Eval] Fallback routing complete. "
        f"Replaced {len(fallback_tasks)} evaluator(s)."
    )

    return updated_results, updated_models


def _get_model_type(model) -> str:
    """Infer model_type string from a model client instance."""
    from ..model_client import (
        QwenOmniClient, GeminiClient, OpenAICompatibleClient,
    )
    if isinstance(model, GeminiClient):
        return "gemini"
    elif isinstance(model, OpenAICompatibleClient):
        return "openai_compatible"
    elif isinstance(model, QwenOmniClient):
        return "qwen_omni"
    return ""


def run_stage2(
    models: List[ModelClient],
    round_name: str,
    video_path: str,
    audio_path: str,
    generated_content: dict,
    references: dict = None,
    events: dict = None,
    shot_timestamps: list = None,
    media_cache=None,
) -> Tuple[str, str]:
    """
    Execute Stage2 multi-model evaluation.

    Each model automatically retries until the API successfully returns a valid evaluation result.
    All models must pass for the result to be considered pass.

    Args:
        models:             List of evaluation model clients (2-3)
        round_name:         Current round name
        video_path:         Video path
        audio_path:         Audio path
        generated_content:  Content to evaluate from Stage1 output
        references:         Previous round's references (context)
        events:             Previous round's events (context)
        shot_timestamps:    Shot boundary list (needed for shot round evaluation)
        media_cache:        MediaCache instance

    Returns:
        (verdict, feedback, eval_details):
          verdict:      "pass"    - all models consider it correct
                        "retry"   - some models found issues (feedback loop)
                        "discard" - all models found issues (discard)
          feedback:     Aggregated feedback string (empty when verdict=="pass")
          eval_details: Complete evaluation result list for each eval model (for intermediate result recording)
    """
    logger.info(
        f"[Stage2-Eval] round={round_name}, num_evaluators={len(models)}"
    )

    # 1. Build prompts
    sys_prompt = get_eval_system_prompt(round_name)
    user_text = get_eval_user_prompt(
        round_name=round_name,
        generated_content=generated_content,
        references=references,
        events=events,
        shot_timestamps=shot_timestamps,
    )

    # 2. Build multimodal content
    if media_cache is not None:
        content = media_cache.build_content_with_text(
            video_path, audio_path, user_text
        )
    else:
        content = build_multimodal_content(
            video_path=video_path,
            audio_path=audio_path,
            text_prompt=user_text,
            use_timestamp_video=True,
        )

    # 3. Call each evaluation model in parallel (each model has internal API retry, waits until success)
    eval_results = [None] * len(models)

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = [
            executor.submit(_eval_single_with_retry, i, model, content, sys_prompt)
            for i, model in enumerate(models)
        ]
        for future in as_completed(futures):
            idx, parsed = future.result()
            eval_results[idx] = parsed

    # 3.5 Smart fallback routing: if any model hit INPUT_TOO_LONG,
    #     replace it with fallback models (gemini_25/gemini_31) and re-evaluate
    eval_results, models = _handle_input_too_long_fallback(
        eval_results=eval_results,
        models=models,
        content=content,
        sys_prompt=sys_prompt,
    )

    # 4. Record evaluation scores (for log analysis and subsequent decision-making)
    for i, r in enumerate(eval_results):
        scores = r.get("criterion_scores", {})
        overall = r.get("overall_score", "N/A")
        logger.info(
            f"[Stage2-Eval] Evaluation model {i} scores: overall={overall}, "
            f"factual={scores.get('factual_accuracy', 'N/A')}, "
            f"temporal={scores.get('temporal_consistency', 'N/A')}, "
            f"structural={scores.get('structural_integrity', 'N/A')}"
        )

    # 5. Voting logic: all models must pass for the result to be considered pass
    total = len(eval_results)
    fail_count = sum(
        1 for r in eval_results if not r.get("is_correct", False)
    )
    pass_count = total - fail_count

    logger.info(
        f"[Stage2-Eval] Voting result: {pass_count}/{total} passed, "
        f"{fail_count}/{total} failed"
    )

    if fail_count == 0:
        # All passed → proceed to Stage3
        return "pass", "", eval_results

    # Aggregate feedback from failed evaluation models
    feedback_parts = []
    for i, r in enumerate(eval_results):
        if not r.get("is_correct", False):
            issues = r.get("issues", [])
            if issues:
                feedback_parts.append(f"--- Evaluator {i} feedback ---")
                for issue in issues:
                    if isinstance(issue, dict):
                        sev = issue.get("severity", "unknown")
                        loc = issue.get("location", "unknown")
                        desc = issue.get("description", "")
                        sug = issue.get("suggestion", "")
                        feedback_parts.append(f"[{sev}] at {loc}: {desc}")
                        if sug:
                            feedback_parts.append(f"  Suggestion: {sug}")
                    else:
                        feedback_parts.append(str(issue))

    feedback = "\n".join(feedback_parts)

    # Voting decision logic (v2):
    #   All pass → pass
    #   Any model passes → retry (with feedback from failed models)
    #   All fail → discard
    if pass_count == 0:
        # All failed → discard the result
        logger.info(f"[Stage2-Eval] DISCARD: all {total} evaluators failed")
        return "discard", feedback, eval_results
    else:
        # Some models passed but not all → feedback loop back to Stage1 for regeneration
        logger.info(f"[Stage2-Eval] RETRY: {pass_count}/{total} passed, {fail_count}/{total} failed — retrying with feedback")
        return "retry", feedback, eval_results


def run_stage2_single(
    models: List[ModelClient],
    round_name: str,
    video_path: str,
    audio_path: str,
    generated_content: dict,
    references: dict = None,
    events: dict = None,
    shot_timestamps: list = None,
    media_cache=None,
) -> bool:
    """
    Simplified Stage2 evaluation, returns True/False.
    Used for re-evaluating refined captions in Stage3.
    Returns True only when all models pass (pass verdict).
    """
    verdict, _, _ = run_stage2(
        models=models,
        round_name=round_name,
        video_path=video_path,
        audio_path=audio_path,
        generated_content=generated_content,
        references=references,
        events=events,
        shot_timestamps=shot_timestamps,
        media_cache=media_cache,
    )
    return verdict == "pass"
