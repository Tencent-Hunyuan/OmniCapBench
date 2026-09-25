#!/usr/bin/env python3
"""
MTSS Eval - standalone LLM-as-Judge local semantic scoring.

This file follows metrics.md exactly:
  - LLM judge is a separate table and is not mixed into rule-based metrics.
  - It only scores structurally aligned local fields.
  - Reported normalized metrics are:
    LLM_person_ref (recall + precision), LLM_scene_ref (recall + precision),
    LLM_event, LLM_subshot, LLM_shot_fact.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import fcntl
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

# Import matching utilities from mtss_metrics
from .mtss_metrics import (
    EPS,
    canonicalize_caption,
    detail_description,
    f1_score,
    load_jsonl,
    match_references,
    match_events,
    match_shots,
    subshots_for,
)
from .mtss_api_clients import UnifiedTextClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mtss_llm_judge")

SHOT_CONTEXT_RADIUS = 2
SUBSHOT_CONTEXT_RADIUS = 2
SUBSHOT_MATCH_DIMS = {"LLM_subshot_match_precision", "LLM_subshot_match_recall"}
SUBSHOT_SCORE_DIMS = {"LLM_subshot_precision", "LLM_subshot_recall"}


# ======================================================================
# System prompt for all judge calls — enforces evidence-based scoring
# ======================================================================

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict, evidence-based evaluator for video description quality. "
    "Your job is to compare a PREDICTION against a GROUND TRUTH and score 1-5. "
    "Use only the fields shown in the prompt; do not infer from outside knowledge or missing video context.\n\n"
    "RULES:\n"
    "1. Judge factual consistency and coverage of the requested field only.\n"
    "2. Contradictions to the GT are severe errors; missing details are less severe than wrong details.\n"
    "3. Ignore wording/order differences when the same fact is expressed.\n"
    "4. Do not evaluate hallucination as a separate criterion. Ignore facts outside the requested coverage checklist; score only factual coverage of the requested field.\n"
    "5. Output format: first write <score>N</score> (N=1-5), then 2-3 concise evidence bullets naming the key matched/missed/wrong facts.\n"
)

_SUBSHOT_MATCH_SYSTEM_PROMPT = (
    "You are a strict binary matcher for video local visual states. "
    "Decide whether the TARGET item has at least one true counterpart in the opposite-side candidate list. "
    "Use only the provided sub_shot_content, times, ids, and reference mapping. "
    "A match must share the main visible subject(s), action/pose/state, and local scene facts. "
    "Accept paraphrases and small granularity differences; reject generic overlap, nearby-but-different actions, or contradictory core facts. "
    "Do not evaluate hallucination or extra details separately; decide only whether the target's core local facts are covered by at least one candidate. "
    "Output exactly <match>1</match> for a match or <match>0</match> for no match, followed by one short reason."
)

_SUBSHOT_QUALITY_SYSTEM_PROMPT = (
    "You are an evidence-based evaluator for local video-state fact coverage. "
    "Compare the TARGET local-state facts against the provided matched context and score 1-5. "
    "Do not evaluate hallucination or extra details as a separate criterion; "
    "score only whether the requested TARGET facts are covered. "
    "Ignore minor wording differences and neighboring context that is not relevant to the target. "
    "Output format: first write <score>N</score> (N=1-5), then 2-3 short evidence bullets naming the hit/missed target facts."
)


# ======================================================================
# LLM Judge Client
# ======================================================================

class LLMJudgeClient:
    """Unified API client for LLM-as-Judge scoring."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 max_tokens: int = 512, temperature: float = 0.1,
                 timeout: int = 300, max_retries: int = 5, **kwargs):
        self.model = model
        # Set env vars for reference matching in mtss_metrics
        os.environ["REF_MATCH_BASE_URL"] = str(base_url)
        os.environ["REF_MATCH_MODEL"] = str(model)
        os.environ["REF_MATCH_API_KEY"] = str(api_key)
        os.environ["REF_MATCH_MAX_TOKENS"] = str(max_tokens)
        os.environ["REF_MATCH_TEMPERATURE"] = str(temperature)
        os.environ["REF_MATCH_TIMEOUT"] = str(timeout)
        os.environ["REF_MATCH_MAX_RETRIES"] = str(max_retries)
        self.client = UnifiedTextClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    def score(self, prompt: str, system_prompt: str = None) -> Optional[dict]:
        """
        Send a scoring prompt to the LLM and parse the response.
        Returns dict with 'score' (int 1-5) and 'reasoning' (str), or None on failure.
        """
        content = self.client.complete(prompt, system_prompt=system_prompt)
        return self._parse_score(content) if content else None

    def binary_match(self, prompt: str, system_prompt: str = None) -> Optional[dict]:
        """Send a binary matching prompt and parse <match>0/1</match>."""
        content = self.client.complete(prompt, system_prompt=system_prompt)
        return self._parse_binary_match(content) if content else None

    @staticmethod
    def _parse_score(text: str) -> dict:
        """Parse LLM response to extract score (1-5) and reasoning."""
        # Try <score>N</score> pattern (primary format)
        m = re.search(r'<score>\s*([1-5])\s*</score>', text)
        if m:
            return {"score": int(m.group(1)), "reasoning": text.strip()}

        # Try "Score: N" or "Score = N" pattern
        m = re.search(r'[Ss]core\s*[:=]\s*([1-5])(?:\s|\.|/|$)', text)
        if m:
            return {"score": int(m.group(1)), "reasoning": text.strip()}

        # Fallback: look for N/5 pattern
        m = re.search(r'([1-5])\s*/\s*5', text)
        if m:
            return {"score": int(m.group(1)), "reasoning": text.strip()}

        # Last resort: default to 3 with parse failure note
        return {"score": 3, "reasoning": f"[PARSE_FAILED] {text[:300]}"}

    @staticmethod
    def _parse_binary_match(text: str) -> dict:
        """Parse LLM response to extract a strict binary match decision."""
        m = re.search(r'<match>\s*([01])\s*</match>', text)
        if m:
            return {"match": int(m.group(1)), "reasoning": text.strip()}
        m = re.search(r'\b(match|matched|yes|true)\b', text, flags=re.IGNORECASE)
        n = re.search(r'\b(no match|not matched|no|false)\b', text, flags=re.IGNORECASE)
        if n and not m:
            return {"match": 0, "reasoning": text.strip()}
        if m and not n:
            return {"match": 1, "reasoning": text.strip()}
        return {"match": 0, "reasoning": f"[PARSE_FAILED] {text[:300]}"}


# ======================================================================
# Prompt Templates for Each Atomic Dimension
# ======================================================================

def build_reference_recall_prompt(gt_ref: dict, pred_ref: dict) -> str:
    """Build RECALL prompt: GT as subject, check if prediction covers all GT facts."""
    gt_type = gt_ref.get("type", "unknown")
    if gt_type == "person":
        field_guidance = (
            "For PERSON entities, the detail_description typically includes: "
            "gender, approximate age range, ethnicity/skin tone, hairstyle and hair color, "
            "facial features (face shape, glasses, facial hair, scars/moles), "
            "body build/height, clothing (top/bottom/shoes/accessories with colors and patterns), "
            "held objects, and any other distinguishing visual traits. "
            "Each of these factual elements counts as a separate coverage point."
        )
    elif gt_type == "scene":
        field_guidance = (
            "For SCENE entities, the detail_description typically includes: "
            "location type (indoor/outdoor, specific venue), spatial layout and dimensions, "
            "lighting conditions (natural/artificial, brightness, color temperature), "
            "dominant colors and textures of surfaces (walls, floor, ceiling), "
            "key furniture/fixtures/objects present in the environment, "
            "atmosphere/weather conditions if outdoor, signage or text visible, "
            "and any other distinguishing environmental traits. "
            "Each of these factual elements counts as a separate coverage point."
        )
    else:
        field_guidance = (
            "Focus on all concrete visual/physical attributes mentioned in the detail_description. "
            "Each distinct factual element counts as a separate coverage point."
        )
    return (
        f"Entity type: {gt_type}\n\n"
        "TASK: Evaluate RECALL — using the GROUND TRUTH as the subject, check whether the PREDICTION "
        "covers ALL factual content present in the GT detail_description. "
        "Do not evaluate hallucination or extra prediction details; only score coverage of GT facts.\n\n"
        "Compare only appearance_anchor.id_features.detail_description. "
        "Ignore semantic_description, attributes, timestamp, events, shots, and context.\n\n"
        f"{field_guidance}\n\n"
        "Scoring criteria (recall perspective — does prediction cover GT facts?):\n"
        "- Count the total number of distinct factual elements in the GT description.\n"
        "- For each GT fact, check if the prediction mentions it (same meaning, wording differences OK).\n"
        "- A GT fact is 'missed' if the prediction does not mention it at all.\n"
        "- A GT fact is 'contradicted' if the prediction states the opposite (worse than missing).\n\n"
        "=== GROUND TRUTH detail_description (SUBJECT — all facts here should be covered) ===\n"
        f"{detail_description(gt_ref)}\n\n"
        "=== PREDICTION detail_description (check coverage of GT facts) ===\n"
        f"{detail_description(pred_ref)}\n\n"
        "Score 1-5:\n"
        "  1 = Prediction covers almost none of the GT facts (<20%), or contradicts the core identity\n"
        "  2 = Prediction covers a minority of GT facts (20-40%); multiple key attributes are missing\n"
        "  3 = Prediction covers roughly half of GT facts (40-70%); some important details are missing\n"
        "  4 = Prediction covers most GT facts (70-90%); only minor details are missing\n"
        "  5 = Prediction covers essentially all GT facts (>90%); at most trivial omissions\n"
    )


def build_reference_precision_prompt(gt_ref: dict, pred_ref: dict) -> str:
    """Build PRECISION prompt: Prediction as subject, check if GT covers all prediction facts."""
    gt_type = gt_ref.get("type", "unknown")
    if gt_type == "person":
        field_guidance = (
            "For PERSON entities, the detail_description typically includes: "
            "gender, approximate age range, ethnicity/skin tone, hairstyle and hair color, "
            "facial features (face shape, glasses, facial hair, scars/moles), "
            "body build/height, clothing (top/bottom/shoes/accessories with colors and patterns), "
            "held objects, and any other distinguishing visual traits. "
            "Each of these factual elements counts as a separate verification point."
        )
    elif gt_type == "scene":
        field_guidance = (
            "For SCENE entities, the detail_description typically includes: "
            "location type (indoor/outdoor, specific venue), spatial layout and dimensions, "
            "lighting conditions (natural/artificial, brightness, color temperature), "
            "dominant colors and textures of surfaces (walls, floor, ceiling), "
            "key furniture/fixtures/objects present in the environment, "
            "atmosphere/weather conditions if outdoor, signage or text visible, "
            "and any other distinguishing environmental traits. "
            "Each of these factual elements counts as a separate verification point."
        )
    else:
        field_guidance = (
            "Focus on all concrete visual/physical attributes mentioned in the detail_description. "
            "Each distinct factual element counts as a separate verification point."
        )
    return (
        f"Entity type: {gt_type}\n\n"
        "TASK: Evaluate PRECISION — using the PREDICTION as the subject, check whether the GT "
        "covers ALL factual content present in the prediction detail_description. "
        "Do not evaluate hallucination as a separate criterion; only score coverage of prediction facts by the GT.\n\n"
        "Compare only appearance_anchor.id_features.detail_description. "
        "Ignore semantic_description, attributes, timestamp, events, shots, and context.\n\n"
        f"{field_guidance}\n\n"
        "Scoring criteria (precision perspective — are prediction facts supported by GT?):\n"
        "- Count the total number of distinct factual elements in the PREDICTION description.\n"
        "- For each prediction fact, check if the GT mentions or supports it.\n"
        "- A prediction fact is 'not covered' if the GT does not mention the same factual content.\n"
        "- A prediction fact is 'contradicted' if the GT states the opposite; count it as not covered.\n\n"
        "=== GROUND TRUTH detail_description (reference for verification) ===\n"
        f"{detail_description(gt_ref)}\n\n"
        "=== PREDICTION detail_description (SUBJECT — all facts here should be verified) ===\n"
        f"{detail_description(pred_ref)}\n\n"
        "Score 1-5:\n"
        "  1 = GT covers almost none of the prediction facts (<20%)\n"
        "  2 = GT covers a minority of prediction facts (20-40%)\n"
        "  3 = GT covers roughly half of prediction facts (40-70%)\n"
        "  4 = GT covers most prediction facts (70-90%)\n"
        "  5 = GT covers essentially all prediction facts (>90%)\n"
    )


def build_event_quality_prompt(gt_event: dict, pred_event: dict,
                               gt_refs: List[dict], pred_refs: List[dict],
                               ref_mapping: Dict[str, str]) -> str:
    """Build prompt for matched non-dialogue event content.description only."""
    gt_type = gt_event.get("type", "unknown")
    gt_content = gt_event.get("content", {}) if isinstance(gt_event.get("content"), dict) else {}
    pred_content = pred_event.get("content", {}) if isinstance(pred_event.get("content"), dict) else {}
    ref_context = _build_ref_context(gt_refs, pred_refs, ref_mapping)
    return (
        f"Event type: {gt_type}\n\n"
        f"{ref_context}\n\n"
        "Compare only content.description. Do not judge dialogue line, speaker field, event type, "
        "time_range, shot placement, or reference ID correctness except as context for the description.\n\n"
        "An event description typically captures: the main action/activity happening, "
        "which entities (persons/objects) are involved and their roles, "
        "spatial relationships between entities, cause-effect or sequential sub-actions, "
        "environmental/audio context relevant to the event, and the outcome or state change. "
        "Each of these factual elements is a separate coverage point.\n\n"
        "Scoring approach:\n"
        "- Identify all distinct factual claims in the GT description.\n"
        "- Check which are present in the prediction (same meaning, wording differences OK).\n"
        "- Contradictions (stating the opposite) are worse than omissions.\n"
        "- Do not evaluate hallucination or extra prediction details; ignore facts outside the GT checklist.\n\n"
        "=== GROUND TRUTH content.description ===\n"
        f"{gt_content.get('description', '')}\n\n"
        "=== PREDICTION content.description ===\n"
        f"{pred_content.get('description', '')}\n\n"
        "Score 1-5:\n"
        "  1 = Prediction contradicts or entirely misses the core event action/subject (<20% GT facts covered)\n"
        "  2 = Prediction captures the broad category but misses or contradicts major factual elements (20-40%)\n"
        "  3 = Prediction captures the main gist but omits or gets wrong several important details (40-70%)\n"
        "  4 = Prediction covers most GT facts accurately; only minor details are missing or imprecise (70-90%)\n"
        "  5 = Prediction faithfully covers essentially all GT factual content (>90%)\n"
    )


def _format_subshot_group(subshots: List[dict], focus_key: str = "") -> str:
    lines = []
    for sub in subshots:
        if not isinstance(sub, dict):
            continue
        key = sub.get("_sub_key") or sub.get("_sub_id") or sub.get("sub_shot_id", "")
        mark = " [TARGET]" if focus_key and key == focus_key else ""
        lines.append(
            f"- id={key}{mark}; shot={sub.get('_shot_id', '')}; "
            f"time={sub.get('sub_shot_time', [])}: {sub.get('sub_shot_content', '')}"
        )
    return "\n".join(lines) or "(empty)"


def build_subshot_quality_prompt(gt_subshots: List[dict], pred_subshots: List[dict],
                                 gt_refs: List[dict], pred_refs: List[dict],
                                 ref_mapping: Dict[str, str],
                                 direction: str = "recall",
                                 target_key: str = "") -> str:
    """Build prompt for matched local visual-state sub_shot_content only."""
    ref_context = _build_ref_context(gt_refs, pred_refs, ref_mapping)
    gt_text = _format_subshot_group(gt_subshots, target_key if direction == "recall" else "")
    pred_text = _format_subshot_group(pred_subshots, target_key if direction == "precision" else "")
    if direction == "precision":
        scoring_text = (
            "Score the TARGET PREDICTION subshot against the GT context (PRECISION direction). "
            "Judge only whether the target prediction's concrete local-state facts are supported by or accurately hit the GT context. "
            "Do not evaluate hallucination or extra target details separately; only score coverage of the target prediction facts by the GT context. "
            "A target fact that conflicts with, or fails to match, the GT context is not covered.\n\n"
            "A sub_shot_content typically describes: which entities are visible (persons/objects with ref IDs), "
            "their current pose/action/gesture/expression, spatial positions and relationships between entities, "
            "interaction with objects or environment, motion direction/speed, "
            "and any transient visual state (e.g., lighting change, object appearing/disappearing). "
            "Each of these factual elements is a separate verification point."
        )
    else:
        scoring_text = (
            "Score how well the prediction context covers the TARGET GT subshot facts (RECALL direction). "
            "Judge only the GT local-state facts that should be recovered. "
            "Do not evaluate hallucination or prediction-side extra details; only score coverage of the target GT facts. "
            "A GT target fact that is omitted or contradicted by the prediction context is not covered.\n\n"
            "A sub_shot_content typically describes: which entities are visible (persons/objects with ref IDs), "
            "their current pose/action/gesture/expression, spatial positions and relationships between entities, "
            "interaction with objects or environment, motion direction/speed, "
            "and any transient visual state (e.g., lighting change, object appearing/disappearing). "
            "Each of these factual elements is a separate coverage point."
        )
    return (
        f"{ref_context}\n\n"
        "Compare only local visual state text: sub_shot_content. "
        "Use ids/times only to understand which item is the target and nearby context; do not score timing accuracy here.\n\n"
        f"{scoring_text}\n\n"
        "Scoring approach:\n"
        "- Identify all distinct factual claims in the TARGET item.\n"
        "- For each fact, check if the opposite-side context contains it (same meaning, wording OK).\n"
        "- Contradictions count as not covered.\n"
        "- Nearby context items help disambiguate but are not themselves scored.\n"
        "- Do not evaluate hallucination or extra details outside the target checklist.\n\n"
        "=== GROUND TRUTH sub_shot_content context ===\n"
        f"{gt_text}\n\n"
        "=== PREDICTION sub_shot_content context ===\n"
        f"{pred_text}\n\n"
        "Score 1-5:\n"
        "  1 = Target local state is essentially not hit; core subject/action/state is absent or contradicted (<20% facts covered)\n"
        "  2 = A small fraction of target facts are hit (20-40%), but core subject/action/state is missing or wrong\n"
        "  3 = Core state is partially hit (40-70%), with important target facts missing or inaccurate\n"
        "  4 = Most target facts are hit (70-90%); only minor misses or imprecision\n"
        "  5 = Target facts are fully and accurately hit (>90%); at most trivial omissions\n"
    )


def build_subshot_match_prompt(gt_subshots: List[dict], pred_subshots: List[dict],
                               gt_refs: List[dict], pred_refs: List[dict],
                               ref_mapping: Dict[str, str],
                               direction: str = "recall",
                               target_key: str = "") -> str:
    """Build binary LLM prompt for bidirectional subshot matching."""
    ref_context = _build_ref_context(gt_refs, pred_refs, ref_mapping)
    gt_text = _format_subshot_group(gt_subshots, target_key if direction == "recall" else "")
    pred_text = _format_subshot_group(pred_subshots, target_key if direction == "precision" else "")
    task_text = (
        "Task: decide whether the TARGET PREDICTION subshot matches at least one GT candidate in the context."
        if direction == "precision"
        else "Task: decide whether at least one prediction candidate matches the TARGET GT subshot."
    )
    return (
        f"{ref_context}\n\n"
        f"{task_text} "
        "A match should preserve the main entities, visible state, action/pose, and local scene facts. "
        "Use [TARGET] to identify the item being evaluated. "
        "Ignore minor wording differences and slight granularity shifts. "
        "Reject candidates that only share a broad scene, are merely adjacent in time, or contradict the target's core subject/action/state. "
        "Do not evaluate hallucination or extra details; this binary decision is only about whether the target facts are covered by at least one candidate.\n\n"
        "=== GROUND TRUTH local visual states ===\n"
        f"{gt_text}\n\n"
        "=== PREDICTION local visual states ===\n"
        f"{pred_text}\n\n"
        "Output exactly one decision:\n"
        "  <match>1</match> if the target has at least one true counterpart in the opposite context\n"
        "  <match>0</match> if no candidate matches, or the best candidate is only generic/adjacent\n"
    )


def build_shot_transfer_prompt(gt_shot: dict, pred_shot: dict,
                               gt_refs: List[dict], pred_refs: List[dict],
                               ref_mapping: Dict[str, str]) -> str:
    """Build prompt to evaluate matched shot camera + shot_transfer only."""
    gt_transfer = gt_shot.get("shot_transfer", "")
    pred_transfer = pred_shot.get("shot_transfer", "")
    gt_camera = gt_shot.get("camera", {})
    pred_camera = pred_shot.get("camera", {})
    if not gt_transfer and not gt_camera:
        return ""
    return (
        "Compare only camera and shot_transfer fields. Do not judge visual_description, active_events, "
        "references_in_shot, event content, or timing.\n\n"
        "Camera facts include: shot scale (close-up/medium/wide/extreme), camera angle (high/low/eye-level/bird's-eye/dutch), "
        "camera movement (pan/tilt/dolly/zoom/tracking/crane/handheld/static), "
        "framing composition (rule-of-thirds/centered/off-center), depth of field (shallow/deep), "
        "focus subject, and any special techniques (rack focus, whip pan, etc.).\n"
        "Shot_transfer facts include: transition type (cut/dissolve/fade/wipe/match-cut), "
        "transition speed, and any visual continuity or discontinuity noted.\n\n"
        "Scoring approach:\n"
        "- Identify all distinct camera/transfer facts in the GT.\n"
        "- Check which are present in the prediction (same meaning, terminology differences OK).\n"
        "- Contradictions are worse than omissions.\n"
        "- Do not evaluate hallucination or extra prediction details; ignore facts outside the GT camera/transfer checklist.\n\n"
        "=== GROUND TRUTH ===\n"
        f"camera: {json.dumps(gt_camera, ensure_ascii=False)}\n"
        f"shot_transfer: {gt_transfer}\n\n"
        "=== PREDICTION ===\n"
        f"camera: {json.dumps(pred_camera, ensure_ascii=False)}\n"
        f"shot_transfer: {pred_transfer}\n\n"
        "Score 1-5:\n"
        "  1 = Camera and transfer facts are mostly missing or contradicted (<20% GT facts covered)\n"
        "  2 = Some overlap but major camera/transfer facts are wrong or missing (20-40%)\n"
        "  3 = Partly correct; some key camera/transfer details are missing or inaccurate (40-70%)\n"
        "  4 = Mostly accurate with only minor omissions or imprecision (70-90%)\n"
        "  5 = Fully consistent with all GT camera and shot_transfer facts (>90%)\n"
    )


def _build_ref_context(gt_refs: List[dict], pred_refs: List[dict],
                       ref_mapping: Dict[str, str]) -> str:
    """Build a reference context string so the LLM understands ref_id mappings."""
    lines = ["Reference ID Mapping (Pred → GT):"]
    for pred_id, gt_id in ref_mapping.items():
        gt_ref = next((r for r in gt_refs if r.get("ref_id") == gt_id), None)
        pred_ref = next((r for r in pred_refs if r.get("ref_id") == pred_id), None)
        gt_desc = detail_description(gt_ref) if gt_ref else "?"
        pred_desc = detail_description(pred_ref) if pred_ref else "?"
        lines.append(f"  {pred_id} ({pred_desc}) → {gt_id} ({gt_desc})")
    return "\n".join(lines) if len(lines) > 1 else ""


def _flatten_subshots(shots: List[dict]) -> List[dict]:
    flat = []
    for shot_idx, shot in enumerate(shots):
        shot_id = str(shot.get("shot_id", f"shot_{shot_idx}"))
        for sub_idx, sub in enumerate(subshots_for(shot)):
            item = dict(sub)
            sub_id = str(item.get("_sub_id") or item.get("sub_shot_id") or f"sub_{sub_idx}")
            item["_shot_index"] = shot_idx
            item["_shot_id"] = shot_id
            item["_sub_index"] = sub_idx
            item["_global_sub_index"] = len(flat)
            item["_sub_key"] = f"{shot_id}|{sub_id}"
            flat.append(item)
    return flat


def _subshots_in_shot_window(flat_subshots: List[dict], center_shot_idx: int,
                             radius: int = SHOT_CONTEXT_RADIUS) -> List[dict]:
    lo = max(0, center_shot_idx - radius)
    hi = center_shot_idx + radius
    return [
        sub
        for sub in flat_subshots
        if lo <= int(sub.get("_shot_index", -1)) <= hi
    ]


def _projected_opposite_subshot_group(
    flat_subshots: List[dict],
    center_shot_idx: int,
    source_sub_idx: int,
    shot_radius: int = SHOT_CONTEXT_RADIUS,
    subshot_radius: int = SUBSHOT_CONTEXT_RADIUS,
) -> List[dict]:
    ordered = _subshots_in_shot_window(flat_subshots, center_shot_idx, shot_radius)
    if not ordered:
        return []

    center_shot_subs = [
        sub for sub in ordered
        if int(sub.get("_shot_index", -1)) == center_shot_idx
    ]
    if not center_shot_subs:
        return []

    center_in_shot = min(max(int(source_sub_idx), 0), len(center_shot_subs) - 1)
    center_key = center_shot_subs[center_in_shot].get("_sub_key")
    center_pos = next(
        (idx for idx, sub in enumerate(ordered) if sub.get("_sub_key") == center_key),
        None,
    )
    if center_pos is None:
        return []
    lo = max(0, center_pos - subshot_radius)
    hi = min(len(ordered), center_pos + subshot_radius + 1)
    return ordered[lo:hi]


def _subshot_neighbor_group(flat_subshots: List[dict], target: dict,
                            radius: int = SUBSHOT_CONTEXT_RADIUS) -> List[dict]:
    center = int(target.get("_global_sub_index", -1))
    if center < 0:
        return [target]
    lo = max(0, center - radius)
    hi = min(len(flat_subshots), center + radius + 1)
    return flat_subshots[lo:hi]


def _build_subshot_match_and_score_tasks(
    gt_shots: List[dict],
    pred_shots: List[dict],
    shot_matches: List[Tuple[int, int, float]],
    gt_refs: List[dict],
    pred_refs: List[dict],
    ref_mapping: Dict[str, str],
    video_path: str,
) -> Tuple[List[dict], dict]:
    gt_flat = _flatten_subshots(gt_shots)
    pred_flat = _flatten_subshots(pred_shots)
    gt_by_key = {sub["_sub_key"]: sub for sub in gt_flat}
    pred_by_key = {sub["_sub_key"]: sub for sub in pred_flat}
    pred_to_gt_shot = {pi: gi for pi, gi, _ in shot_matches}
    gt_to_pred_shot = {gi: pi for pi, gi, _ in shot_matches}
    tasks: List[dict] = []

    for pred_sub in pred_flat:
        pred_shot_idx = int(pred_sub.get("_shot_index", -1))
        if pred_shot_idx not in pred_to_gt_shot:
            continue
        gt_candidates = _projected_opposite_subshot_group(
            gt_flat,
            pred_to_gt_shot[pred_shot_idx],
            int(pred_sub.get("_sub_index", 0)),
        )
        if not gt_candidates:
            continue
        pred_key = pred_sub["_sub_key"]
        pred_group = _subshot_neighbor_group(pred_flat, pred_sub)
        tasks.append({
            "dimension": "LLM_subshot_match_precision",
            "video_path": video_path,
            "gt_id": pred_key,
            "pred_id": pred_key,
            "target_key": pred_key,
            "pred_sub_keys": [pred_key],
            "candidate_gt_keys": [sub["_sub_key"] for sub in gt_candidates],
            "prompt": build_subshot_match_prompt(
                gt_candidates, [pred_sub], gt_refs, pred_refs, ref_mapping,
                direction="precision", target_key=pred_key,
            ),
        })
        tasks.append({
            "dimension": "LLM_subshot_precision",
            "video_path": video_path,
            "gt_id": pred_key,
            "pred_id": pred_key,
            "target_key": pred_key,
            "requires_match_dimension": "LLM_subshot_match_precision",
            "requires_match_id": pred_key,
            "prompt": build_subshot_quality_prompt(
                gt_candidates, pred_group, gt_refs, pred_refs, ref_mapping,
                direction="precision", target_key=pred_key,
            ),
        })

    for gt_sub in gt_flat:
        gt_shot_idx = int(gt_sub.get("_shot_index", -1))
        if gt_shot_idx not in gt_to_pred_shot:
            continue
        pred_candidates = _projected_opposite_subshot_group(
            pred_flat,
            gt_to_pred_shot[gt_shot_idx],
            int(gt_sub.get("_sub_index", 0)),
        )
        if not pred_candidates:
            continue
        gt_key = gt_sub["_sub_key"]
        gt_group = _subshot_neighbor_group(gt_flat, gt_sub)
        tasks.append({
            "dimension": "LLM_subshot_match_recall",
            "video_path": video_path,
            "gt_id": gt_key,
            "pred_id": "",
            "target_key": gt_key,
            "gt_sub_keys": [gt_key],
            "candidate_pred_keys": [sub["_sub_key"] for sub in pred_candidates],
            "prompt": build_subshot_match_prompt(
                [gt_sub], pred_candidates, gt_refs, pred_refs, ref_mapping,
                direction="recall", target_key=gt_key,
            ),
        })
        tasks.append({
            "dimension": "LLM_subshot_recall",
            "video_path": video_path,
            "gt_id": gt_key,
            "pred_id": "",
            "target_key": gt_key,
            "requires_match_dimension": "LLM_subshot_match_recall",
            "requires_match_id": gt_key,
            "prompt": build_subshot_quality_prompt(
                gt_group, pred_candidates, gt_refs, pred_refs, ref_mapping,
                direction="recall", target_key=gt_key,
            ),
        })

    counts = {
        "video_path": video_path,
        "num_gt_subshots": len(gt_by_key),
        "num_pred_subshots": len(pred_by_key),
    }
    return tasks, counts


# ======================================================================
# Scoring Pipeline: Collect all scoring tasks for one video
# ======================================================================

def collect_video_scoring_context(gt_item: dict, pred_item: dict) -> Tuple[List[dict], dict]:
    """
    Collect metrics.md LLM judge tasks for one video.
    Only matched/aligned records are scored.
    """
    gt_caption, _ = canonicalize_caption(gt_item.get("mtss_caption", {}))
    pred_caption, _ = canonicalize_caption(pred_item.get("model_prediction", {}))

    gt_refs = gt_caption["references"]
    pred_refs = pred_caption["references"]
    gt_events = gt_caption["events"]
    pred_events = pred_caption["events"]
    gt_shots = gt_caption["shots"]
    pred_shots = pred_caption["shots"]

    ref_mapping = match_references(gt_refs, pred_refs)
    event_pack = match_events(gt_events, pred_events)
    shot_pack = match_shots(gt_shots, pred_shots)

    tasks = []
    video_path = gt_item.get("video_path", "")

    pred_ref_by_id = {str(r.get("ref_id", "")).upper(): r for r in pred_refs}
    gt_ref_by_id = {str(r.get("ref_id", "")).upper(): r for r in gt_refs}
    for pred_id, gt_id in ref_mapping.items():
        gt_ref = gt_ref_by_id.get(gt_id)
        pred_ref = pred_ref_by_id.get(pred_id)
        if not gt_ref or not pred_ref:
            continue
        ref_type = str(gt_ref.get("type", "")).lower()
        if ref_type not in {"person", "animal", "object", "scene"}:
            continue
        # Paper groups person+animal+object as "Subject", scene as "Scene"
        dim_prefix = "LLM_person_ref" if ref_type in ("person", "animal", "object") else "LLM_scene_ref"
        # Recall: GT as subject, does prediction cover all GT content?
        tasks.append({
            "dimension": f"{dim_prefix}_recall",
            "video_path": video_path,
            "gt_id": gt_id,
            "pred_id": pred_id,
            "ref_type": ref_type,
            "prompt": build_reference_recall_prompt(gt_ref, pred_ref),
        })
        # Precision: Prediction as subject, does GT cover all prediction content?
        tasks.append({
            "dimension": f"{dim_prefix}_precision",
            "video_path": video_path,
            "gt_id": gt_id,
            "pred_id": pred_id,
            "ref_type": ref_type,
            "prompt": build_reference_precision_prompt(gt_ref, pred_ref),
        })

    for pi, gi, score in event_pack["matches"]:
        gt_ev = gt_events[gi]
        pred_ev = pred_events[pi]
        if str(gt_ev.get("type", "")).lower() == "dialogue":
            continue
        tasks.append({
            "dimension": "LLM_event",
            "video_path": video_path,
            "gt_id": gt_ev.get("event_id", ""),
            "pred_id": pred_ev.get("event_id", ""),
            "event_type": gt_ev.get("type", "unknown"),
            "match_score": score,
            "prompt": build_event_quality_prompt(gt_ev, pred_ev, gt_refs, pred_refs, ref_mapping),
        })

    subshot_tasks, subshot_counts = _build_subshot_match_and_score_tasks(
        gt_shots, pred_shots, shot_pack["matches"], gt_refs, pred_refs, ref_mapping, video_path
    )
    tasks.extend(subshot_tasks)

    for pi, gi, score in shot_pack["matches"]:
        gt_shot = gt_shots[gi]
        pred_shot = pred_shots[pi]
        prompt = build_shot_transfer_prompt(gt_shot, pred_shot, gt_refs, pred_refs, ref_mapping)
        if not prompt:
            continue
        tasks.append({
            "dimension": "LLM_shot_fact",
            "video_path": video_path,
            "gt_id": gt_shot.get("shot_id", ""),
            "pred_id": pred_shot.get("shot_id", ""),
            "match_score": score,
            "prompt": prompt,
        })

    return tasks, subshot_counts


def collect_scoring_tasks(gt_item: dict, pred_item: dict) -> List[dict]:
    """Backward-compatible task collector."""
    tasks, _ = collect_video_scoring_context(gt_item, pred_item)
    return tasks


# ======================================================================
# Batch Scoring
# ======================================================================

def run_llm_judge(gt_path: str, pred_path: str, output_path: str,
                  client: LLMJudgeClient, workers: int = 32,
                  limit: int = 0) -> Dict[str, Any]:
    """
    Run LLM-as-Judge scoring on all matched pairs.
    """
    gt_data = load_jsonl(gt_path)
    pred_data = load_jsonl(pred_path)

    # Build lookup
    pred_lookup = {}
    for item in pred_data:
        vp = item.get("video_path", "")
        if vp and item.get("status") == "success":
            pred_lookup[vp] = item

    logger.info(f"GT: {len(gt_data)}, Pred: {len(pred_lookup)} successful")

    # Collect all scoring tasks (parallel ref matching via LLM)
    pairs = []
    for gt_item in gt_data:
        vp = gt_item.get("video_path", "")
        if vp not in pred_lookup:
            continue
        pairs.append((gt_item, pred_lookup[vp]))
        if limit > 0 and len(pairs) >= limit:
            break

    def _collect_one(pair):
        gt_item, pred_item = pair
        vp = gt_item.get("video_path", "")
        try:
            tasks, subshot_counts = collect_video_scoring_context(gt_item, pred_item)
            return tasks, subshot_counts, vp, None
        except Exception as e:
            return [], {}, vp, e

    all_tasks = []
    subshot_totals = {}
    video_count = 0
    skipped_videos = 0
    collect_workers = max(1, workers)
    logger.info("Collecting scoring tasks from %d videos with %d workers...", len(pairs), collect_workers)

    total_pairs = len(pairs)
    done_count = 0
    log_interval = max(1, total_pairs // 20)
    with ThreadPoolExecutor(max_workers=collect_workers) as pool:
        futures = {pool.submit(_collect_one, p): i for i, p in enumerate(pairs)}
        indexed = {}
        for future in as_completed(futures):
            idx = futures[future]
            tasks, subshot_counts, vp, err = future.result()
            if err:
                skipped_videos += 1
                logger.warning(f"collect_scoring_tasks failed for {os.path.basename(vp)}: {err}")
            else:
                indexed[idx] = (tasks, subshot_counts, vp)
                video_count += 1
            done_count += 1
            if done_count % log_interval == 0 or done_count == total_pairs:
                logger.info("Collecting tasks: %d/%d (%.0f%%)", done_count, total_pairs, done_count / total_pairs * 100)
    for i in sorted(indexed):
        tasks, subshot_counts, vp = indexed[i]
        all_tasks.extend(tasks)
        subshot_totals[vp] = subshot_counts

    if skipped_videos:
        logger.warning(f"Skipped {skipped_videos} videos due to scoring task collection errors")
    logger.info(f"Collected {len(all_tasks)} scoring tasks from {video_count} videos")
    logger.info(f"  LLM_person_ref_recall:    {sum(1 for t in all_tasks if t['dimension'] == 'LLM_person_ref_recall')}")
    logger.info(f"  LLM_person_ref_precision: {sum(1 for t in all_tasks if t['dimension'] == 'LLM_person_ref_precision')}")
    logger.info(f"  LLM_scene_ref_recall:     {sum(1 for t in all_tasks if t['dimension'] == 'LLM_scene_ref_recall')}")
    logger.info(f"  LLM_scene_ref_precision:  {sum(1 for t in all_tasks if t['dimension'] == 'LLM_scene_ref_precision')}")
    logger.info(f"  LLM_event:      {sum(1 for t in all_tasks if t['dimension'] == 'LLM_event')}")
    logger.info(f"  LLM_subshot_match_precision: {sum(1 for t in all_tasks if t['dimension'] == 'LLM_subshot_match_precision')}")
    logger.info(f"  LLM_subshot_match_recall:    {sum(1 for t in all_tasks if t['dimension'] == 'LLM_subshot_match_recall')}")
    logger.info(f"  LLM_subshot_precision:       {sum(1 for t in all_tasks if t['dimension'] == 'LLM_subshot_precision')}")
    logger.info(f"  LLM_subshot_recall:          {sum(1 for t in all_tasks if t['dimension'] == 'LLM_subshot_recall')}")
    logger.info(f"  LLM_shot_fact:  {sum(1 for t in all_tasks if t['dimension'] == 'LLM_shot_fact')}")

    detail_path = output_path.replace(".json", "_detail.jsonl")
    os.makedirs(os.path.dirname(detail_path) or ".", exist_ok=True)

    def _task_key(task: dict) -> str:
        return f"{task.get('dimension')}|{task.get('video_path')}|{task.get('gt_id')}"

    def _load_done_keys() -> set:
        done = set()
        if not os.path.exists(detail_path):
            return done
        with open(detail_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add(f"{d.get('dimension')}|{d.get('video_path')}|{d.get('gt_id')}")
        return done

    def _load_successful_match_keys() -> set:
        success = set()
        if not os.path.exists(detail_path):
            return success
        with open(detail_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("dimension") in SUBSHOT_MATCH_DIMS and int(d.get("match", 0) or 0) == 1:
                    success.add(f"{d.get('dimension')}|{d.get('video_path')}|{d.get('gt_id')}")
        return success

    def _score_one(task):
        dim = task.get("dimension")
        if dim in SUBSHOT_MATCH_DIMS:
            result = client.binary_match(task["prompt"], system_prompt=_SUBSHOT_MATCH_SYSTEM_PROMPT)
        elif dim in SUBSHOT_SCORE_DIMS:
            result = client.score(task["prompt"], system_prompt=_SUBSHOT_QUALITY_SYSTEM_PROMPT)
        else:
            result = client.score(task["prompt"], system_prompt=_JUDGE_SYSTEM_PROMPT)
        return task, result

    def _run_tasks(tasks: List[dict], label: str) -> Tuple[int, int]:
        done_keys = _load_done_keys()
        pending_tasks = [task for task in tasks if _task_key(task) not in done_keys]
        logger.info("%s pending: %d", label, len(pending_tasks))
        if not pending_tasks:
            return 0, 0

        scored = 0
        failed = 0
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_score_one, t): t for t in pending_tasks}
            batch_results = []
            for future in as_completed(futures):
                task, result = future.result()
                record = {k: v for k, v in task.items() if k != "prompt"}

                if result and "match" in result:
                    record["match"] = int(result["match"])
                    record["reasoning"] = result["reasoning"][:300]
                    scored += 1
                elif result:
                    record["score"] = result["score"]
                    record["reasoning"] = result["reasoning"][:300]
                    scored += 1
                else:
                    record["score"] = None
                    record["reasoning"] = "LLM scoring failed"
                    failed += 1

                batch_results.append(record)
                if len(batch_results) >= 50:
                    _append_results(detail_path, batch_results)
                    batch_results = []

                total_done = scored + failed
                if total_done % 100 == 0 or total_done == len(pending_tasks):
                    elapsed = time.time() - start_time
                    rate = total_done / max(elapsed, 1)
                    eta = (len(pending_tasks) - total_done) / max(rate, 0.01)
                    sys.stderr.write(
                        f"\r  {label}: {total_done}/{len(pending_tasks)} "
                        f"({total_done/len(pending_tasks)*100:.1f}%) "
                        f"| ok {scored} fail {failed} "
                        f"| {rate:.1f}/s | ETA {eta:.0f}s  "
                    )
                    sys.stderr.flush()

            if batch_results:
                _append_results(detail_path, batch_results)
        sys.stderr.write("\n")
        logger.info("%s complete: %d scored, %d failed", label, scored, failed)
        return scored, failed

    gated_subshot_score_tasks = [t for t in all_tasks if t.get("dimension") in SUBSHOT_SCORE_DIMS]
    first_pass_tasks = [t for t in all_tasks if t.get("dimension") not in SUBSHOT_SCORE_DIMS]
    _run_tasks(first_pass_tasks, "First-pass judge")

    successful_matches = _load_successful_match_keys()
    eligible_subshot_score_tasks = [
        task for task in gated_subshot_score_tasks
        if (
            f"{task.get('requires_match_dimension')}|{task.get('video_path')}|{task.get('requires_match_id')}"
            in successful_matches
        )
    ]
    skipped_scores = len(gated_subshot_score_tasks) - len(eligible_subshot_score_tasks)
    logger.info("Subshot semantic score tasks gated by match=1: %d eligible, %d skipped",
                len(eligible_subshot_score_tasks), skipped_scores)
    _run_tasks(eligible_subshot_score_tasks, "Subshot semantic judge")

    return _aggregate_results(detail_path, output_path, subshot_totals=subshot_totals)


def _append_results(detail_path: str, results: List[dict]):
    """Append results to detail JSONL with file locking."""
    lock_path = detail_path + ".lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            with open(detail_path, "a", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def _aggregate_results(detail_path: str, output_path: str,
                       subshot_totals: Optional[Dict[str, dict]] = None) -> Dict[str, Any]:
    """Aggregate per-task 1-5 scores into metrics.md normalized LLM judge table."""
    all_scores = defaultdict(list)
    per_video = defaultdict(lambda: defaultdict(list))
    subshot_totals = subshot_totals or {}
    subshot_match_records = []
    score_records = []

    if not os.path.exists(detail_path):
        os.makedirs(os.path.dirname(detail_path) or ".", exist_ok=True)
        open(detail_path, "a", encoding="utf-8").close()

    with open(detail_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            dim = d.get("dimension", "unknown")
            if dim in SUBSHOT_MATCH_DIMS:
                if d.get("match") is not None:
                    subshot_match_records.append(d)
                continue
            score_records.append(d)

    successful_matches = {
        f"{d.get('dimension')}|{d.get('video_path')}|{d.get('gt_id')}"
        for d in subshot_match_records
        if int(d.get("match", 0) or 0) == 1
    }
    for d in score_records:
        dim = d.get("dimension", "unknown")
        if dim == "LLM_subshot":
            continue
        if dim in SUBSHOT_SCORE_DIMS:
            match_key = f"{d.get('requires_match_dimension')}|{d.get('video_path')}|{d.get('requires_match_id')}"
            if match_key not in successful_matches:
                continue
            score = d.get("score")
        else:
            score = d.get("score")
        if score is None:
            continue
        all_scores[dim].append(score)
        per_video[d.get("video_path", "")][dim].append(score)

    def norm_mean(scores: List[int]) -> Optional[float]:
        if not scores:
            return None
        return float((np.mean(scores) - 1) / 4)

    subshot_scores = all_scores.get("LLM_subshot_precision", []) + all_scores.get("LLM_subshot_recall", [])
    # Compute per-direction reference scores
    person_recall = norm_mean(all_scores.get("LLM_person_ref_recall", []))
    person_precision = norm_mean(all_scores.get("LLM_person_ref_precision", []))
    scene_recall = norm_mean(all_scores.get("LLM_scene_ref_recall", []))
    scene_precision = norm_mean(all_scores.get("LLM_scene_ref_precision", []))

    def mean_present(values: List[Optional[float]]) -> Optional[float]:
        present = [v for v in values if v is not None]
        return float(np.mean(present)) if present else None

    # Compact: mean of recall and precision when available.
    person_ref = mean_present([person_recall, person_precision])
    scene_ref = mean_present([scene_recall, scene_precision])
    metrics = {
        "LLM_person_ref": person_ref,
        "LLM_person_ref_recall": person_recall,
        "LLM_person_ref_precision": person_precision,
        "LLM_scene_ref": scene_ref,
        "LLM_scene_ref_recall": scene_recall,
        "LLM_scene_ref_precision": scene_precision,
        "LLM_event": norm_mean(all_scores.get("LLM_event", [])),
        "LLM_subshot": norm_mean(subshot_scores),
        "LLM_subshot_precision": norm_mean(all_scores.get("LLM_subshot_precision", [])),
        "LLM_subshot_recall": norm_mean(all_scores.get("LLM_subshot_recall", [])),
        "LLM_shot_fact": norm_mean(all_scores.get("LLM_shot_fact", [])),
    }

    total_gt_subshots = 0
    total_pred_subshots = 0
    matched_gt_keys = set()
    matched_pred_keys = set()
    for video_path, counts in subshot_totals.items():
        num_gt = int(counts.get("num_gt_subshots", 0) or 0)
        num_pred = int(counts.get("num_pred_subshots", 0) or 0)
        total_gt_subshots += num_gt
        total_pred_subshots += num_pred
    for rec in subshot_match_records:
        if int(rec.get("match", 0) or 0) != 1:
            continue
        video_path = rec.get("video_path", "")
        if rec.get("dimension") == "LLM_subshot_match_precision":
            key = rec.get("target_key") or rec.get("pred_id") or rec.get("gt_id")
            matched_pred_keys.add((video_path, key))
        elif rec.get("dimension") == "LLM_subshot_match_recall":
            key = rec.get("target_key") or rec.get("gt_id")
            matched_gt_keys.add((video_path, key))

    llm_subshot_recall = (
        len(matched_gt_keys) / (total_gt_subshots + EPS)
        if total_gt_subshots
        else None
    )
    llm_subshot_precision = (
        len(matched_pred_keys) / (total_pred_subshots + EPS)
        if total_pred_subshots
        else (None if total_gt_subshots == 0 else 0.0)
    )
    llm_subshot_f1 = (
        f1_score(llm_subshot_precision, llm_subshot_recall)
        if llm_subshot_precision is not None and llm_subshot_recall is not None
        else None
    )

    summary = {
        "metric_version": "mtss_llm_judge_metrics_md_v1",
        "total_tasks_scored": sum(len(v) for v in all_scores.values()) + len(subshot_match_records),
        "per_video_count": len(set(per_video.keys()) | set(subshot_totals.keys())),
        "llm_judge_table": metrics,
        "llm_subshot_local_matching": {
            "Subshot_Recall": llm_subshot_recall,
            "Subshot_Precision": llm_subshot_precision,
            "Subshot_F1": llm_subshot_f1,
            "num_gt_subshots": total_gt_subshots,
            "num_pred_subshots": total_pred_subshots,
            "num_matched_gt_subshots": len(matched_gt_keys),
            "num_matched_pred_subshots": len(matched_pred_keys),
            "num_llm_match_tasks": len(subshot_match_records),
            "num_llm_precision_match_tasks": sum(1 for r in subshot_match_records if r.get("dimension") == "LLM_subshot_match_precision"),
            "num_llm_recall_match_tasks": sum(1 for r in subshot_match_records if r.get("dimension") == "LLM_subshot_match_recall"),
        },
        "per_dimension": {},
    }

    dimension_order = [
        "LLM_person_ref", "LLM_person_ref_recall", "LLM_person_ref_precision",
        "LLM_scene_ref", "LLM_scene_ref_recall", "LLM_scene_ref_precision",
        "LLM_event",
        "LLM_subshot", "LLM_subshot_precision", "LLM_subshot_recall",
        "LLM_shot_fact",
    ]
    for dim in dimension_order:
        if dim == "LLM_subshot":
            scores = subshot_scores
        elif dim == "LLM_person_ref":
            scores = all_scores.get("LLM_person_ref_recall", []) + all_scores.get("LLM_person_ref_precision", [])
        elif dim == "LLM_scene_ref":
            scores = all_scores.get("LLM_scene_ref_recall", []) + all_scores.get("LLM_scene_ref_precision", [])
        else:
            scores = all_scores.get(dim, [])
        if not scores:
            continue
        arr = np.array(scores, dtype=float)
        summary["per_dimension"][dim] = {
            "count": len(scores),
            "mean_1to5": float(np.mean(arr)),
            "normalized_mean": metrics[dim],
            "std_1to5": float(np.std(arr)),
            "median_1to5": float(np.median(arr)),
            "score_distribution": {str(i): int(np.sum(arr == i)) for i in range(1, 6)},
        }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to: {output_path}")
    return summary


def print_judge_summary(summary: Dict[str, Any]):
    """Pretty-print metrics.md LLM judge summary."""
    print("\n" + "=" * 70)
    print("  MTSS Standalone LLM-as-Judge")
    print("=" * 70)
    print(f"  Total tasks scored: {summary.get('total_tasks_scored', 0)}")
    print(f"  Videos evaluated:   {summary.get('per_video_count', 0)}")
    print()
    print("  LLM Judge Table (normalized to 0-1)")
    print("  ─────────────────────────────────────────────────────────")
    table = summary.get("llm_judge_table", {})
    for key in [
        "LLM_person_ref", "LLM_person_ref_recall", "LLM_person_ref_precision",
        "LLM_scene_ref", "LLM_scene_ref_recall", "LLM_scene_ref_precision",
        "LLM_event",
        "LLM_subshot", "LLM_subshot_precision", "LLM_subshot_recall",
        "LLM_shot_fact",
    ]:
        value = table.get(key)
        if value is None:
            print(f"    {key:<30s} N/A")
        else:
            bar_len = int(max(0.0, min(1.0, value)) * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            count = summary.get("per_dimension", {}).get(key, {}).get("count", 0)
            print(f"    {key:<30s} {bar} {value:.4f} (n={count})")
    match = summary.get("llm_subshot_local_matching", {})
    if match:
        print()
        print("  LLM Subshot Local Matching (binary match, F1 compact)")
        print("  ─────────────────────────────────────────────────────────")
        for key in ["Subshot_Recall", "Subshot_Precision", "Subshot_F1"]:
            value = match.get(key)
            if value is None:
                print(f"    {key:<24s} N/A")
            else:
                print(f"    {key:<24s} {value:.4f}")
        print(
            f"    matched_gt/pred           "
            f"{match.get('num_matched_gt_subshots', 0)}/{match.get('num_gt_subshots', 0)} GT, "
            f"{match.get('num_matched_pred_subshots', 0)}/{match.get('num_pred_subshots', 0)} Pred"
        )
    print("=" * 70)


# ======================================================================
# CLI Entry Point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="MTSS LLM-as-Judge Description Quality Scoring")
    parser.add_argument("--gt", type=str, required=True,
                        help="Path to GT JSONL")
    parser.add_argument("--pred", type=str, required=True,
                        help="Path to prediction JSONL")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for judge summary")
    parser.add_argument("--base-url", type=str,
                        default=os.environ.get("OMNICAP_EVAL_JUDGE_BASE_URL", "http://localhost:8000/v1"),
                        help="LLM API base URL (any OpenAI-compatible endpoint)")
    parser.add_argument("--model", type=str,
                        default=os.environ.get("OMNICAP_EVAL_JUDGE_MODEL", "gpt-4o"),
                        help="Model name for the judge")
    parser.add_argument("--api-key", type=str,
                        default=os.environ.get("OMNICAP_EVAL_JUDGE_API_KEY", "EMPTY"),
                        help="API key")
    parser.add_argument("--max-tokens", type=int,
                        default=int(os.environ.get("OMNICAP_EVAL_JUDGE_MAX_TOKENS", "4096")),
                        help="Max output tokens for judge")
    parser.add_argument("--temperature", type=float,
                        default=float(os.environ.get("OMNICAP_EVAL_JUDGE_TEMPERATURE", "0.7")),
                        help="Sampling temperature for judge")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-request timeout")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of videos (0=all)")
    args = parser.parse_args()

    # Auto-detect output path
    if args.output is None:
        pred_dir = os.path.dirname(args.pred)
        args.output = os.path.join(pred_dir, "llm_judge_scores.json")

    # Create client
    client = LLMJudgeClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
    )

    # Test connectivity
    logger.info(f"Judge model: {args.model} @ {args.base_url}")
    try:
        test_resp = requests.get(
            f"{args.base_url.rstrip('/')}/models", timeout=15
        )
        if test_resp.status_code == 200:
            models = [m["id"] for m in test_resp.json().get("data", [])]
            logger.info(f"API reachable. Models: {models[:5]}")
        else:
            logger.warning(f"API returned HTTP {test_resp.status_code}")
    except Exception as e:
        logger.warning(f"API connectivity test failed: {e}")
        logger.warning("Proceeding anyway — will retry on actual requests.")

    # Run scoring
    summary = run_llm_judge(
        args.gt, args.pred, args.output,
        client, workers=args.workers, limit=args.limit
    )
    print_judge_summary(summary)


if __name__ == "__main__":
    main()
