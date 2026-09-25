#!/usr/bin/env python3
"""MTSS Pipeline - Stage4: Atomic-Unit Selection
Stage4 atomic unit selection phase.
Selects the optimal combination at the atomic unit level from up to 3 candidate captions.
The Selector only makes selections, without any modifications.
Includes business-level retry: handles empty responses and parse failures caused by API instability.

Candidate sources:
  - Candidate A: original verified caption before refinement
  - Candidate B: refined caption (model 1)
  - Candidate C: refined caption (model 2), optional

Atomic unit granularity:
  - references: each reference object
  - events: each event object
  - shots: visual_description / camera within each shot can be selected independently
"""
import json
import logging
import time as _time

from .. import config
from ..model_client import ModelClient, parse_response_to_json
from ..media_utils import build_multimodal_content
from ..prompts.stage4_select import get_select_system_prompt, get_select_user_prompt

logger = logging.getLogger("omnicapbench_pipeline")

# Business-level retry for Stage4 Select (handles API instability)
_SELECT_BIZ_MAX_ATTEMPTS = 5
_SELECT_BIZ_RETRY_DELAY = 5.0


def run_stage4(
    model: ModelClient,
    round_name: str,
    video_path: str,
    audio_path: str,
    candidate_a: dict,
    candidate_b: dict,
    candidate_c: dict = None,
    references: dict = None,
    events: dict = None,
    media_cache=None,
) -> dict:
    """
    Execute Stage4 atomic unit selection.

    Includes a business-level retry loop (up to 5 attempts), ensuring that API instability
    does not directly lose the opportunity for selection optimization.

    Args:
        model:        Selection model client
        round_name:   Current round name
        video_path:   Video path
        audio_path:   Audio path
        candidate_a:  Original verified caption (before refinement)
        candidate_b:  Refined caption 1 (Stage3 model 1)
        candidate_c:  Refined caption 2 (Stage3 model 2), optional
        references:   Previous round's references (context)
        events:       Previous round's events (context)
        media_cache:  MediaCache instance

    Returns:
        Final selected/assembled content dict.
        Falls back to candidate_a when all retries are exhausted.
    """
    num_candidates = 3 if candidate_c is not None else 2
    logger.info(f"[Stage4-Select] round={round_name}, candidates={num_candidates}")

    sys_prompt = get_select_system_prompt(round_name)
    user_text = get_select_user_prompt(
        round_name=round_name,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        candidate_c=candidate_c,
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

    target_map = {
        "reference_bank": "references",
        "event": "events",
        "shot": "shots",
    }
    target_key = target_map.get(round_name, round_name)
    final_key = f"final_{target_key}"

    # Business-level retry loop
    for attempt in range(_SELECT_BIZ_MAX_ATTEMPTS):
        try:
            raw = model.inference(content, system_prompt=sys_prompt)

            if not raw:
                logger.warning(
                    f"[Stage4-Select] Empty response "
                    f"(biz attempt {attempt+1}/{_SELECT_BIZ_MAX_ATTEMPTS}), retrying..."
                )
                if attempt < _SELECT_BIZ_MAX_ATTEMPTS - 1:
                    _time.sleep(min(_SELECT_BIZ_RETRY_DELAY * (1.5 ** attempt), 30.0))
                continue

            # Log raw response for debug
            logger.debug(
                f"[Stage4-Select] raw response (length={len(raw)}):\n"
                f"{raw}"
            )

            parsed = parse_response_to_json(raw)
            if parsed.get("parse_error"):
                logger.warning(
                    f"[Stage4-Select] JSON parse failed "
                    f"(biz attempt {attempt+1}/{_SELECT_BIZ_MAX_ATTEMPTS}), "
                    f"raw (length={len(raw)}):\n{raw[:500]}"
                )
                if attempt < _SELECT_BIZ_MAX_ATTEMPTS - 1:
                    _time.sleep(min(_SELECT_BIZ_RETRY_DELAY * (1.5 ** attempt), 30.0))
                continue

            # Extract final assembled result
            if final_key in parsed:
                result = {target_key: parsed[final_key]}
                logger.info(
                    f"[Stage4-Select] Selection complete (attempt {attempt+1}), "
                    f"assembled {len(parsed[final_key])} elements"
                )
                return result
            elif target_key in parsed:
                # Model directly output the target array
                logger.info(
                    f"[Stage4-Select] Model directly output the target array (attempt {attempt+1})"
                )
                return parsed
            elif "selections" in parsed:
                # Try to assemble from selections
                logger.info(
                    f"[Stage4-Select] Manually assembling from selections (attempt {attempt+1})"
                )
                return _assemble_from_selections(
                    parsed["selections"], candidate_a, candidate_b,
                    round_name, candidate_c=candidate_c,
                )
            else:
                # Unrecognized output format — retry
                logger.warning(
                    f"[Stage4-Select] '{final_key}' or 'selections' key not found, "
                    f"got keys={list(parsed.keys())} "
                    f"(biz attempt {attempt+1}/{_SELECT_BIZ_MAX_ATTEMPTS}), retrying..."
                )
                if attempt < _SELECT_BIZ_MAX_ATTEMPTS - 1:
                    _time.sleep(min(_SELECT_BIZ_RETRY_DELAY * (1.5 ** attempt), 30.0))
                continue

        except Exception as e:
            logger.warning(
                f"[Stage4-Select] Exception on biz attempt "
                f"{attempt+1}/{_SELECT_BIZ_MAX_ATTEMPTS}: {e}"
            )
            if attempt < _SELECT_BIZ_MAX_ATTEMPTS - 1:
                _time.sleep(min(_SELECT_BIZ_RETRY_DELAY * (1.5 ** attempt), 30.0))
            continue

    # All attempts exhausted — fallback to candidate_a
    logger.error(
        f"[Stage4-Select] All {_SELECT_BIZ_MAX_ATTEMPTS} business-level attempts failed, "
        f"fallback to candidate A"
    )
    print(f"    ⚠️  Stage4 Select failed after {_SELECT_BIZ_MAX_ATTEMPTS} attempts, using candidate A")
    return candidate_a


def _assemble_from_selections(
    selections: list,
    candidate_a: dict,
    candidate_b: dict,
    round_name: str,
    candidate_c: dict = None,
) -> dict:
    """
    Manually assemble the final output based on selection results.
    Each selection: {"element_id": ..., "chosen": "A"|"B"|"C", "sub_selections": {...}}
    """
    target_map = {
        "reference_bank": "references",
        "event": "events",
        "shot": "shots",
    }
    target_key = target_map.get(round_name, round_name)

    arr_a = candidate_a.get(target_key, [])
    arr_b = candidate_b.get(target_key, [])
    arr_c = candidate_c.get(target_key, []) if candidate_c else []

    if not arr_a and not arr_b:
        return candidate_a

    # Build index by element ID
    id_field_map = {
        "references": "ref_id",
        "events": "event_id",
        "shots": "shot_id",
    }
    id_field = id_field_map.get(target_key, "id")

    lookup_a = {item.get(id_field, f"idx_{i}"): item
                for i, item in enumerate(arr_a)}
    lookup_b = {item.get(id_field, f"idx_{i}"): item
                for i, item in enumerate(arr_b)}
    lookup_c = {item.get(id_field, f"idx_{i}"): item
                for i, item in enumerate(arr_c)} if arr_c else {}

    # Build selection mapping
    choice_map = {}
    sub_sel_map = {}
    for sel in selections:
        eid = sel.get("element_id", "")
        chosen = sel.get("chosen", "A").upper()
        choice_map[eid] = chosen
        if "sub_selections" in sel:
            sub_sel_map[eid] = sel["sub_selections"]

    # Assemble result
    result = []
    for i, item_a in enumerate(arr_a):
        eid = item_a.get(id_field, f"idx_{i}")
        chosen = choice_map.get(eid, "A")

        if target_key == "shots" and eid in sub_sel_map:
            # Shot level: supports independent sub-component selection
            sub = sub_sel_map[eid]
            item_b = lookup_b.get(eid, item_a)
            item_c = lookup_c.get(eid, item_a) if lookup_c else item_a
            assembled = dict(item_a)  # Use A as the base

            # visual_description independent selection
            vd_choice = sub.get("visual_description", "A").upper()
            if vd_choice == "B":
                assembled["visual_description"] = item_b.get(
                    "visual_description", item_a.get("visual_description", [])
                )
            elif vd_choice == "C" and lookup_c:
                assembled["visual_description"] = item_c.get(
                    "visual_description", item_a.get("visual_description", [])
                )
            # Also supports fine-grained sub_shot level selection
            elif isinstance(vd_choice, list):
                orig_vd_a = item_a.get("visual_description", [])
                orig_vd_b = item_b.get("visual_description", [])
                orig_vd_c = item_c.get("visual_description", []) if lookup_c else []
                new_vd = []
                for k, choice in enumerate(vd_choice):
                    c = str(choice).upper()
                    if c == "C" and k < len(orig_vd_c):
                        new_vd.append(orig_vd_c[k])
                    elif c == "B" and k < len(orig_vd_b):
                        new_vd.append(orig_vd_b[k])
                    elif k < len(orig_vd_a):
                        new_vd.append(orig_vd_a[k])
                assembled["visual_description"] = new_vd

            # camera independent selection
            cam_choice = sub.get("camera", "A").upper()
            if cam_choice == "B":
                assembled["camera"] = item_b.get(
                    "camera", item_a.get("camera", {})
                )
            elif cam_choice == "C" and lookup_c:
                assembled["camera"] = item_c.get(
                    "camera", item_a.get("camera", {})
                )

            # Immutable fields always remain consistent (taken from A)
            for immutable_field in ["shot_id", "time_range", "active_events", "references_in_shot"]:
                if immutable_field in item_a:
                    assembled[immutable_field] = item_a[immutable_field]

            result.append(assembled)
        elif chosen == "B" and eid in lookup_b:
            result.append(lookup_b[eid])
        elif chosen == "C" and eid in lookup_c:
            result.append(lookup_c[eid])
        else:
            result.append(item_a)

    logger.info(f"[Stage4-Select] Manually assembled {len(result)} elements")
    return {target_key: result}
