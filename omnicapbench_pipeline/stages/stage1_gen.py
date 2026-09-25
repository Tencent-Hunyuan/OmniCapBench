#!/usr/bin/env python3
"""
MTSS Pipeline - Stage1: Data Generation
Stage1 data generation phase.
Based on the current round's input (audio + video + prompt + previous round output), generate key MTSS content.
Supports feedback injection: when Stage2 evaluation fails, inject feedback as context to regenerate.
Supports media_cache: avoids redundant preprocessing of the same video.
Includes business-level retry: retries empty responses and JSON parse failures within Stage1, avoiding wasted Stage2 API calls.
"""

import json
import logging
import time as _time

from .. import config
from ..model_client import ModelClient, parse_response_to_json
from ..media_utils import build_multimodal_content
from ..prompts.stage1_gen import get_gen_system_prompt, get_gen_user_prompt

logger = logging.getLogger("omnicapbench_pipeline")

# Business-level retry: handles empty responses and JSON parse failures
# before returning to the pipeline (avoids wasting a full Stage1→Stage2 cycle)
_STAGE1_BIZ_MAX_ATTEMPTS = 5
_STAGE1_BIZ_RETRY_DELAY = 5.0


def run_stage1(
    model: ModelClient,
    round_name: str,
    video_path: str,
    audio_path: str,
    shot_timestamps: list = None,
    references: dict = None,
    events: dict = None,
    feedback: str = None,
    media_cache=None,
) -> dict:
    """
    Execute Stage1 generation.

    Includes a business-level retry loop (up to 5 attempts), handling empty responses
    and JSON parse failures, avoiding wasting an entire Stage1→Stage2 cycle due to
    a single API fluctuation.

    Args:
        model:           Generation model client
        round_name:      "reference_bank" | "event" | "shot"
        video_path:      Video file path
        audio_path:      Audio file path (empty string triggers auto-extraction)
        shot_timestamps: Shot boundary list [[start, end], ...]
        references:      Output from the reference_bank round (needed for event/shot rounds)
        events:          Output from the event round (needed for shot round)
        feedback:        Feedback string from Stage2 eval (injected during retry)
        media_cache:     MediaCache instance (avoids redundant preprocessing)

    Returns:
        Parsed JSON dict, e.g. {"references": [...]} or {"events": [...]} or {"shots": [...]}
        On failure returns {"parse_error": True, ...}
    """
    logger.info(
        f"[Stage1-Gen] round={round_name}, feedback={'YES' if feedback else 'NO'}"
    )

    # 1. Build system prompt (includes previous round output + feedback + shot_timestamps)
    sys_prompt = get_gen_system_prompt(
        round_name=round_name,
        references=references,
        events=events,
        shot_timestamps=shot_timestamps,
        feedback=feedback,
    )

    # 2. Build user prompt text
    has_audio = bool(audio_path)
    user_text = get_gen_user_prompt(
        round_name=round_name,
        has_audio=has_audio,
        shot_timestamps=shot_timestamps,
    )

    # 3. Build multimodal content (video + audio + text)
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

    # 4. Business-level retry loop: handles empty responses and JSON parse failures
    last_raw = ""
    for attempt in range(_STAGE1_BIZ_MAX_ATTEMPTS):
        try:
            raw_response = model.inference(content, system_prompt=sys_prompt)

            if not raw_response:
                logger.warning(
                    f"[Stage1-Gen] Model returned empty response "
                    f"(biz attempt {attempt+1}/{_STAGE1_BIZ_MAX_ATTEMPTS}), retrying..."
                )
                if attempt < _STAGE1_BIZ_MAX_ATTEMPTS - 1:
                    _time.sleep(min(_STAGE1_BIZ_RETRY_DELAY * (1.5 ** attempt), 30.0))
                continue

            last_raw = raw_response

            # Log raw response for debug
            logger.debug(
                f"[Stage1-Gen] Raw response (length={len(raw_response)}):\n"
                f"{raw_response}"
            )

            # 5. Parse response
            result = parse_response_to_json(raw_response)

            if result.get("parse_error"):
                logger.warning(
                    f"[Stage1-Gen] JSON parse failed "
                    f"(biz attempt {attempt+1}/{_STAGE1_BIZ_MAX_ATTEMPTS}), "
                    f"raw (length={len(raw_response)}):\n{raw_response[:500]}"
                )
                if attempt < _STAGE1_BIZ_MAX_ATTEMPTS - 1:
                    _time.sleep(min(_STAGE1_BIZ_RETRY_DELAY * (1.5 ** attempt), 30.0))
                continue

            # Success
            logger.info(
                f"[Stage1-Gen] Success (attempt {attempt+1}), keys={list(result.keys())}"
            )
            return result

        except Exception as e:
            logger.warning(
                f"[Stage1-Gen] Exception on biz attempt {attempt+1}/{_STAGE1_BIZ_MAX_ATTEMPTS}: {e}"
            )
            if attempt < _STAGE1_BIZ_MAX_ATTEMPTS - 1:
                _time.sleep(min(_STAGE1_BIZ_RETRY_DELAY * (1.5 ** attempt), 30.0))
            continue

    # All business-level attempts exhausted
    logger.error(
        f"[Stage1-Gen] All {_STAGE1_BIZ_MAX_ATTEMPTS} business-level attempts failed"
    )
    return {"parse_error": True, "raw_response": last_raw}
