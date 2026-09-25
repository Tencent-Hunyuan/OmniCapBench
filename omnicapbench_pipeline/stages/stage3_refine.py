#!/usr/bin/env python3
"""
MTSS Pipeline - Stage3: Caption Refinement
Stage3 caption quality refinement phase.
Uses 2 models to refine the content that passed Stage2 evaluation.
Constraint: cannot modify structural parameters such as time_range, ref_id, event_id, shot_id,
PERSON_N identifiers, speaker, line, active_events, references_in_shot, timestamp, etc.

Workflow:
  1. 2 models refine separately → produce caption_A, caption_B
  2. Run Stage2 factual evaluation on A, B respectively
  3. If both pass → proceed to Stage4
  4. If one fails → fall back to Stage1 for regeneration, then go through Stage2 again, iterate
  5. When max_turn is reached:
     - Only one passes → use that one
     - Neither passes → fall back to the un-refined result from Stage2
"""

import logging
import time as _time
from typing import List, Tuple

from .. import config
from ..model_client import ModelClient, parse_response_to_json
from ..media_utils import build_multimodal_content
from ..prompts.stage3_refine import get_refine_system_prompt, get_refine_user_prompt

logger = logging.getLogger("omnicapbench_pipeline")

# Max attempts for a single refine call (handles API instability: empty responses, parse errors, etc.)
_REFINE_MAX_ATTEMPTS = getattr(config, 'API_MAX_RETRIES', 20)
# Max attempts specifically for immutable constraint violations
# (model keeps modifying structural fields — unlikely to fix after many retries)
_REFINE_MAX_CONSTRAINT_VIOLATIONS = 10


def _validate_immutable_fields(original: dict, refined: dict,
                                round_name: str) -> list:
    """
    Programmatically validate that the refined JSON preserves all immutable fields unchanged.
    Returns a list of violations (empty list means pass).

    This is a hard validation that does not depend on model judgment, completely preventing
    the refine step from sneaking changes to structural parameters.
    """
    violations = []
    target_map = {
        "reference_bank": "references",
        "event": "events",
        "shot": "shots",
    }
    target_key = target_map.get(round_name, round_name)

    orig_arr = original.get(target_key, [])
    ref_arr = refined.get(target_key, [])

    # Check if element count is consistent
    if len(orig_arr) != len(ref_arr):
        violations.append(
            f"Element count changed: {len(orig_arr)} -> {len(ref_arr)}"
        )
        return violations  # Count changed, element-wise comparison is meaningless

    id_field_map = {
        "references": "ref_id",
        "events": "event_id",
        "shots": "shot_id",
    }
    id_field = id_field_map.get(target_key, "id")

    for i, (orig_item, ref_item) in enumerate(zip(orig_arr, ref_arr)):
        elem_id = orig_item.get(id_field, f"idx_{i}")

        # General: ID field is immutable
        if orig_item.get(id_field) != ref_item.get(id_field):
            violations.append(
                f"{elem_id}: {id_field} changed "
                f"'{orig_item.get(id_field)}' -> '{ref_item.get(id_field)}'"
            )

        if target_key == "references":
            # type is immutable
            if orig_item.get("type") != ref_item.get("type"):
                violations.append(f"{elem_id}: type changed")
            # timestamp is immutable
            orig_ts = (orig_item.get("appearance_anchor", {})
                       .get("id_features", {}).get("timestamp"))
            ref_ts = (ref_item.get("appearance_anchor", {})
                      .get("id_features", {}).get("timestamp"))
            if orig_ts != ref_ts:
                violations.append(
                    f"{elem_id}: timestamp changed '{orig_ts}' -> '{ref_ts}'"
                )

        elif target_key == "events":
            # type, time_range, speaker, line are immutable
            for field in ["type", "time_range"]:
                if orig_item.get(field) != ref_item.get(field):
                    violations.append(f"{elem_id}: {field} changed")
            orig_content = orig_item.get("content", {})
            ref_content = ref_item.get("content", {})
            for field in ["speaker", "line"]:
                if orig_content.get(field) != ref_content.get(field):
                    violations.append(f"{elem_id}: content.{field} changed")

        elif target_key == "shots":
            # time_range, active_events, references_in_shot, camera are immutable
            for field in ["time_range", "active_events", "references_in_shot", "camera"]:
                if orig_item.get(field) != ref_item.get(field):
                    violations.append(f"{elem_id}: {field} changed")
            # sub_shot_time within the visual_description array is immutable
            orig_vd = orig_item.get("visual_description", [])
            ref_vd = ref_item.get("visual_description", [])
            if isinstance(orig_vd, list) and isinstance(ref_vd, list):
                if len(orig_vd) != len(ref_vd):
                    violations.append(
                        f"{elem_id}: visual_description count changed "
                        f"{len(orig_vd)} -> {len(ref_vd)}"
                    )
                else:
                    for j, (ov, rv) in enumerate(zip(orig_vd, ref_vd)):
                        if isinstance(ov, dict) and isinstance(rv, dict):
                            if ov.get("sub_shot_time") != rv.get("sub_shot_time"):
                                violations.append(
                                    f"{elem_id}: visual_description[{j}].sub_shot_time changed"
                                )

    return violations


def run_stage3_refine_single(
    model: ModelClient,
    round_name: str,
    video_path: str,
    audio_path: str,
    content_to_refine: dict,
    references: dict = None,
    events: dict = None,
    media_cache=None,
) -> dict:
    """
    Execute a single refine using one model, producing one refined caption.
    Includes programmatic immutable field validation; if the refined result violates
    constraints, falls back to the original content.

    Returns:
        Refined content dict; on failure returns the original content.
    """
    logger.info(f"[Stage3-Refine] model={model.tag}, round={round_name}")

    sys_prompt = get_refine_system_prompt(round_name)
    user_text = get_refine_user_prompt(
        round_name=round_name,
        content_to_refine=content_to_refine,
        references=references,
        events=events,
    )

    # Build multimodal content
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

    # Retry loop: handles API instability (empty responses, parse errors, etc.)
    _constraint_violation_count = 0
    for attempt in range(_REFINE_MAX_ATTEMPTS):
        try:
            raw = model.inference(content, system_prompt=sys_prompt)
            if not raw:
                logger.warning(
                    f"[Stage3-Refine] {model.tag} attempt {attempt+1}/{_REFINE_MAX_ATTEMPTS} "
                    f"returned empty response, retrying..."
                )
                if attempt < _REFINE_MAX_ATTEMPTS - 1:
                    _time.sleep(min(3.0 * (1.5 ** attempt), 30.0))
                continue

            # Log raw response for debug
            logger.debug(
                f"[Stage3-Refine] {model.tag} raw response (length={len(raw)}):\n"
                f"{raw}"
            )

            parsed = parse_response_to_json(raw)
            if parsed.get("parse_error"):
                logger.warning(
                    f"[Stage3-Refine] {model.tag} attempt {attempt+1}/{_REFINE_MAX_ATTEMPTS} "
                    f"JSON parse failed, retrying... raw (length={len(raw)}):\n{raw[:500]}"
                )
                if attempt < _REFINE_MAX_ATTEMPTS - 1:
                    _time.sleep(min(3.0 * (1.5 ** attempt), 30.0))
                continue

            # Validate immutable fields (hard check, no model dependency)
            violations = _validate_immutable_fields(content_to_refine, parsed, round_name)
            if violations:
                _constraint_violation_count += 1
                logger.warning(
                    f"[Stage3-Refine] {model.tag} attempt {attempt+1}/{_REFINE_MAX_ATTEMPTS} "
                    f"violated immutable constraints ({len(violations)} issues, "
                    f"violation #{_constraint_violation_count}/{_REFINE_MAX_CONSTRAINT_VIOLATIONS}), retrying..."
                )
                print(f"    \u26a0\ufe0f  {model.tag} refine violated constraints ({len(violations)} issues), retrying...")
                # If we've hit the constraint violation limit, give up
                if _constraint_violation_count >= _REFINE_MAX_CONSTRAINT_VIOLATIONS:
                    logger.error(
                        f"[Stage3-Refine] {model.tag} hit constraint violation limit "
                        f"({_REFINE_MAX_CONSTRAINT_VIOLATIONS}), using original content"
                    )
                    print(f"    \u274c {model.tag} refine hit constraint violation limit ({_REFINE_MAX_CONSTRAINT_VIOLATIONS}), using original")
                    return content_to_refine
                if attempt < _REFINE_MAX_ATTEMPTS - 1:
                    _time.sleep(min(3.0 * (1.5 ** attempt), 30.0))
                continue

            logger.info(f"[Stage3-Refine] {model.tag} refinement OK on attempt {attempt+1}")
            return parsed

        except Exception as e:
            logger.warning(
                f"[Stage3-Refine] {model.tag} attempt {attempt+1}/{_REFINE_MAX_ATTEMPTS} "
                f"exception: {e}"
            )
            if attempt < _REFINE_MAX_ATTEMPTS - 1:
                _time.sleep(min(3.0 * (1.5 ** attempt), 30.0))
            continue
    # All attempts exhausted — fall back to original content
    logger.error(
        f"[Stage3-Refine] {model.tag} all {_REFINE_MAX_ATTEMPTS} attempts failed, "
        f"using original content"
    )
    print(f"    ❌ {model.tag} refine failed after {_REFINE_MAX_ATTEMPTS} attempts, using original")
    return content_to_refine


def run_stage3(
    refine_models: List[ModelClient],
    round_name: str,
    video_path: str,
    audio_path: str,
    verified_content: dict,
    references: dict = None,
    events: dict = None,
    media_cache=None,
) -> Tuple[dict, dict]:
    """
    Execute Stage3 refinement (2 models refine separately).

    Args:
        refine_models:     2 ModelClient instances
        round_name:        Current round
        video_path:        Video path
        audio_path:        Audio path
        verified_content:  Content verified by Stage2
        references:        Previous round's references
        events:            Previous round's events
        media_cache:       MediaCache instance

    Returns:
        (caption_a, caption_b) - two refined versions
    """
    assert len(refine_models) >= 2, "Stage3 requires at least 2 refine models"

    # Run both refine models in parallel for efficiency
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _refine_one(model_idx):
        return run_stage3_refine_single(
            model=refine_models[model_idx],
            round_name=round_name,
            video_path=video_path,
            audio_path=audio_path,
            content_to_refine=verified_content,
            references=references,
            events=events,
            media_cache=media_cache,
        )

    results = [None, None]
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(_refine_one, 0)
        future_b = executor.submit(_refine_one, 1)
        results[0] = future_a.result()
        results[1] = future_b.result()

    return results[0], results[1]
