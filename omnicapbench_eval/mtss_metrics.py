#!/usr/bin/env python3
"""
MTSS rule-based metrics.

This implementation follows metrics.md exactly:
  - SGC is a front gate / structural completeness diagnostic.
  - The main rule-based table reports only 8 compact metric families:
    Ref, RefUse, CCC, WHEN, EVSA, Event, Dialog, Subshot.
  - Appendix/detail output keeps the expanded submetrics.
  - LLM-as-Judge is intentionally kept in mtss_llm_judge.py.
  - No combined rule score, boundary-error diagnostic, order diagnostic,
    prediction-side hallucination rate, event type accuracy, or FAA score is
    part of the report.
"""

import argparse
import json
import logging
import math
import os
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

EPS = 1e-8
REF_THRESHOLD = 0.30
SHOT_IOU_THRESHOLD = 0.30
DIALOGUE_LINE_THRESHOLD = 0.50
EVENT_TIME_THRESHOLD = 0.20
SUBSHOT_WINDOW_DELTA = 1.0
REPORT_METRIC_KEYS = [
    "SGC", "Ref", "RefUse", "CCC", "WHEN", "EVSA", "Event", "Dialog", "Subshot",
    "Ref_Subject_F1", "Ref_Subject_Recall", "Ref_Subject_Precision",
    "Ref_Scene_F1", "Ref_Scene_Recall", "Ref_Scene_Precision",
    "Speaker_F1",
    "ref_recall", "ref_precision", "WHO_shot", "WHO_event", "WHO_sub",
    "CCC_macro", "WHEN_H", "WHEN_E", "WHEN_S",
    "Shot_tIoU", "Event_tIoU", "Subshot_tIoU",
    "shot_recall", "shot_precision", "shot_f1",
    "EVSA_P", "EVSA_R",
    "event_recall", "event_precision", "event_f1",
    "DialogueLineScore", "WHAT_dialogue_sem", "DialogueMergedScore",
    "Subshot_Recall", "Subshot_Precision", "Subshot_F1",
    "num_gt_refs", "num_pred_refs", "num_matched_refs",
    "num_gt_events", "num_pred_events", "num_matched_events",
    "num_gt_shots", "num_pred_shots", "num_matched_shots",
    "num_gt_subshots", "num_pred_subshots",
    "num_matched_gt_subshots", "num_matched_pred_subshots",
    "num_gt_event_shot_links", "num_pred_event_shot_links_mapped",
    "gt_schema_valid", "sgc_score", "sgc_schema_valid",
    "sgc_ref_completeness", "sgc_event_completeness",
    "sgc_shot_completeness", "sgc_subshot_completeness",
    "sgc_link_field_completeness", "sgc_count_coverage",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mtss_metrics")

REF_ID_PATTERN = re.compile(
    r"\[?((?:PERSON|ANIMAL|OBJECT|SCENE|SOURCE|AUDIO)_\d+(?:_(?:STYLE|VIEW)_\d+)?)\]?",
    re.IGNORECASE,
)
EVENT_ID_PATTERN = re.compile(r"\[?((?:EVENT)_\d+)\]?", re.IGNORECASE)


def safe_mean(values: Iterable[Optional[float]], default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else float(default)


def f1_score(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall + EPS)


def as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def valid_range(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        start, end = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if end < start:
        return None
    return [start, end]


def temporal_iou(range_a: Any, range_b: Any) -> float:
    a = valid_range(range_a)
    b = valid_range(range_b)
    if not a or not b:
        return 0.0
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / (union + EPS)


def interval_union_length(ranges: Sequence[Any]) -> float:
    clean = sorted([r for r in (valid_range(x) for x in ranges) if r], key=lambda x: x[0])
    if not clean:
        return 0.0
    merged = [clean[0]]
    for start, end in clean[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def range_to_union_iou(gt_range: Any, pred_ranges: Sequence[Any]) -> float:
    gt = valid_range(gt_range)
    preds = [r for r in (valid_range(x) for x in pred_ranges) if r]
    if not gt or not preds:
        return 0.0
    inter_ranges = [[max(gt[0], p[0]), min(gt[1], p[1])] for p in preds]
    inter_len = interval_union_length([r for r in inter_ranges if r[1] > r[0]])
    union_len = interval_union_length([gt] + preds)
    return inter_len / (union_len + EPS)


def ranges_union_iou(ranges_a: Sequence[Any], ranges_b: Sequence[Any]) -> float:
    clean_a = [r for r in (valid_range(x) for x in ranges_a) if r]
    clean_b = [r for r in (valid_range(x) for x in ranges_b) if r]
    if not clean_a and not clean_b:
        return 1.0
    if not clean_a or not clean_b:
        return 0.0
    intersections = []
    for a_start, a_end in clean_a:
        for b_start, b_end in clean_b:
            start, end = max(a_start, b_start), min(a_end, b_end)
            if end > start:
                intersections.append([start, end])
    inter_len = interval_union_length(intersections)
    union_len = interval_union_length(clean_a + clean_b)
    return inter_len / (union_len + EPS)


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    table = str.maketrans({ch: " " for ch in string.punctuation})
    return " ".join(text.translate(table).split())


def tokenize(text: Any) -> List[str]:
    return normalize_text(text).split()


def rouge_l_f1(a: Any, b: Any) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    dp = [[0] * (len(tb) + 1) for _ in range(len(ta) + 1)]
    for i in range(1, len(ta) + 1):
        for j in range(1, len(tb) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if ta[i - 1] == tb[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    p = lcs / len(tb)
    r = lcs / len(ta)
    return 2 * p * r / (p + r + EPS)


def wer_score(gt_line: Any, pred_line: Any) -> float:
    gt = tokenize(gt_line)
    pred = tokenize(pred_line)
    if not gt:
        return 0.0 if pred else 1.0
    dp = [[0] * (len(pred) + 1) for _ in range(len(gt) + 1)]
    for i in range(len(gt) + 1):
        dp[i][0] = i
    for j in range(len(pred) + 1):
        dp[0][j] = j
    for i in range(1, len(gt) + 1):
        for j in range(1, len(pred) + 1):
            cost = 0 if gt[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    wer = dp[-1][-1] / (len(gt) + EPS)
    return max(0.0, 1.0 - wer)


def detail_description(ref: Dict[str, Any]) -> str:
    anchor = as_dict(ref.get("appearance_anchor"))
    return str(as_dict(anchor.get("id_features")).get("detail_description", "") or "")


def ref_similarity(pred_ref: Dict[str, Any], gt_ref: Dict[str, Any]) -> float:
    return rouge_l_f1(detail_description(pred_ref), detail_description(gt_ref))


def find_ref_ids(value: Any) -> List[str]:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    seen: Set[str] = set()
    refs: List[str] = []
    for match in REF_ID_PATTERN.finditer(text or ""):
        ref_id = match.group(1).upper()
        if ref_id not in seen:
            seen.add(ref_id)
            refs.append(ref_id)
    return refs


def canonicalize_caption(raw: Any) -> Tuple[Dict[str, List[dict]], Dict[str, Any]]:
    diag = {"schema_valid": 0.0, "canonicalization": "ok"}
    if not isinstance(raw, dict):
        diag["canonicalization"] = "not_dict"
        return {"references": [], "events": [], "shots": []}, diag

    cap = dict(raw)
    nested = cap.get("events")
    if not isinstance(nested, list) and isinstance(nested, dict):
        if all(key in nested for key in ("references", "events", "shots")):
            for key in ("references", "events", "shots"):
                cap[key] = nested[key]
            diag["canonicalization"] = "events_nested_caption"

    out = {
        "references": [x for x in as_list(cap.get("references")) if isinstance(x, dict)],
        "events": [x for x in as_list(cap.get("events")) if isinstance(x, dict)],
        "shots": [x for x in as_list(cap.get("shots")) if isinstance(x, dict)],
    }
    diag["schema_valid"] = 1.0 if all(isinstance(cap.get(k), list) for k in ("references", "events", "shots")) else 0.0
    return out, diag


def match_references(gt_refs: List[dict], pred_refs: List[dict], threshold: float = REF_THRESHOLD) -> Dict[str, str]:
    def _ref_id(ref: Dict[str, Any]) -> str:
        return str(ref.get("ref_id", "") or "").upper()

    def _ref_type(ref: Dict[str, Any]) -> str:
        return str(ref.get("type", "") or "").lower()

    def _match_ref_id(value: Any) -> str:
        found = find_ref_ids(value)
        if found:
            return found[0]
        return str(value or "").strip().strip("[]").upper()

    def _items_for_type(refs: List[dict], ref_type: str) -> List[dict]:
        items = []
        for idx, ref in enumerate(refs):
            if _ref_type(ref) != ref_type:
                continue
            rid = _ref_id(ref)
            if not rid:
                continue
            items.append({
                "index": idx,
                "ref_id": rid,
                "detail_description": detail_description(ref),
            })
        return items

    exact_mapping = {
        _ref_id(pred): _ref_id(gt)
        for pred in pred_refs
        for gt in gt_refs
        if _ref_id(pred) and _ref_id(pred) == _ref_id(gt) and _ref_type(pred) == _ref_type(gt)
    }
    if not any(os.environ.get(name) for name in ("REF_MATCH_BASE_URL", "REF_MATCH_API_KEY")):
        return exact_mapping

    def _json_from_response(text: str) -> Any:
        text = str(text or "").strip()
        if not text:
            return None
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        starts = [pos for pos in (text.find("{"), text.find("[")) if pos >= 0]
        if not starts:
            return None
        start = min(starts)
        end = max(text.rfind("}"), text.rfind("]"))
        if end <= start:
            return None
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None

    def _as_match_records(data: Any) -> List[dict]:
        if isinstance(data, dict) and isinstance(data.get("matches"), list):
            return [x for x in data.get("matches", []) if isinstance(x, dict)]
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            records = []
            for pred_id, value in data.items():
                if isinstance(value, dict):
                    rec = dict(value)
                    rec.setdefault("pred_ref_id", pred_id)
                    records.append(rec)
                else:
                    records.append({
                        "pred_ref_id": pred_id,
                        "gt_ref_id": value,
                        "match": 1 if value else 0,
                    })
            return records
        return []

    def _get_client():
        cached = getattr(match_references, "_llm_ref_match_client", None)
        if cached is not None:
            return cached
        try:
            from .mtss_api_clients import UnifiedTextClient
        except Exception as exc:
            logger.warning("LLM ref matcher unavailable: %s", exc)
            return None

        try:
            client = UnifiedTextClient(
                base_url=os.environ.get("REF_MATCH_BASE_URL", "http://localhost:8000/v1"),
                model=os.environ.get("REF_MATCH_MODEL", "gpt-4o"),
                api_key=os.environ.get("REF_MATCH_API_KEY", "EMPTY"),
                temperature=float(os.environ.get("REF_MATCH_TEMPERATURE", "0.0")),
                max_tokens=int(os.environ.get("REF_MATCH_MAX_TOKENS", "4096")),
                timeout=int(os.environ.get("REF_MATCH_TIMEOUT", "600")),
                max_retries=int(os.environ.get("REF_MATCH_MAX_RETRIES", "5")),
            )
        except Exception as exc:
            logger.warning("LLM ref matcher client init failed: %s", exc)
            return None
        logger.info("LLM ref matcher initialized: base_url=%s, model=%s",
                    client.base_url, client.model)
        setattr(match_references, "_llm_ref_match_client", client)
        return client

    def _build_prompt(ref_type: str, gt_items: List[dict], pred_items: List[dict]) -> str:
        return (
            "You are matching MTSS reference entities using only "
            "appearance_anchor.id_features.detail_description.\n"
            f"Reference type: {ref_type}\n\n"
            "Match GT and Prediction items one-to-one. A prediction may match at most one GT item, "
            "and each GT item may be used at most once. Use semantic equivalence of the detail descriptions; "
            "ignore wording/order differences. Reject generic overlap, empty/insufficient descriptions, "
            "or contradictory identity/appearance/scene/object details.\n\n"
            "For every prediction item, output match=1 with exactly one gt_ref_id if it has a true counterpart; "
            "otherwise output match=0 and an empty gt_ref_id. Do not create one-to-many matches.\n\n"
            "=== GT ITEMS ===\n"
            f"{json.dumps(gt_items, ensure_ascii=False, indent=2)}\n\n"
            "=== PREDICTION ITEMS ===\n"
            f"{json.dumps(pred_items, ensure_ascii=False, indent=2)}\n\n"
            "Return only valid JSON in this exact shape:\n"
            "{\n"
            "  \"matches\": [\n"
            "    {\"pred_ref_id\": \"PERSON_1\", \"gt_ref_id\": \"PERSON_1\", \"match\": 1},\n"
            "    {\"pred_ref_id\": \"PERSON_2\", \"gt_ref_id\": \"\", \"match\": 0}\n"
            "  ]\n"
            "}\n"
        )

    client = _get_client()
    if client is None:
        return {}

    system_prompt = (
        "You are a strict binary matcher for structured video reference descriptions. "
        "Use only the provided JSON lists. Enforce one-to-one matching. "
        "Return valid JSON only, with match values exactly 0 or 1."
    )
    MAX_REF_MATCH_RETRIES = 50

    def _match_one_type(ref_type: str) -> List[Tuple[str, str]]:
        gt_items = _items_for_type(gt_refs, ref_type)
        pred_items = _items_for_type(pred_refs, ref_type)
        if not gt_items or not pred_items:
            return []
        gt_ids = {item["ref_id"] for item in gt_items}
        pred_ids = {item["ref_id"] for item in pred_items}
        prompt = _build_prompt(ref_type, gt_items, pred_items)

        for attempt in range(1, MAX_REF_MATCH_RETRIES + 1):
            try:
                response = client.complete(prompt, system_prompt=system_prompt)
            except Exception as exc:
                logger.warning("LLM ref matching failed for type=%s (attempt %d/%d): %s",
                               ref_type, attempt, MAX_REF_MATCH_RETRIES, exc)
                if attempt < MAX_REF_MATCH_RETRIES:
                    import time
                    time.sleep(min(2 ** attempt, 16))
                continue

            records = _as_match_records(_json_from_response(response or ""))
            if not records:
                logger.warning("LLM ref matching returned no parseable matches for type=%s (attempt %d/%d)",
                               ref_type, attempt, MAX_REF_MATCH_RETRIES)
                if attempt < MAX_REF_MATCH_RETRIES:
                    import time
                    time.sleep(min(2 ** attempt, 16))
                continue

            pairs: List[Tuple[str, str]] = []
            local_pred: Set[str] = set()
            local_gt: Set[str] = set()
            for rec in records:
                try:
                    match_value = int(rec.get("match", 0) or 0)
                except (TypeError, ValueError):
                    match_value = 0
                if match_value != 1:
                    continue
                pred_id = _match_ref_id(rec.get("pred_ref_id") or rec.get("pred_id") or rec.get("prediction_ref_id") or "")
                gt_id = _match_ref_id(rec.get("gt_ref_id") or rec.get("gt_id") or rec.get("ground_truth_ref_id") or "")
                if not pred_id and rec.get("pred_index") is not None:
                    try:
                        pred_id = pred_items[int(rec.get("pred_index"))]["ref_id"]
                    except (TypeError, ValueError, IndexError):
                        pred_id = ""
                if not gt_id and rec.get("gt_index") is not None:
                    try:
                        gt_id = gt_items[int(rec.get("gt_index"))]["ref_id"]
                    except (TypeError, ValueError, IndexError):
                        gt_id = ""
                if pred_id not in pred_ids or gt_id not in gt_ids:
                    continue
                if pred_id in local_pred or gt_id in local_gt:
                    continue
                pairs.append((pred_id, gt_id))
                local_pred.add(pred_id)
                local_gt.add(gt_id)
            return pairs

        logger.error("LLM ref matching exhausted all %d retries for type=%s", MAX_REF_MATCH_RETRIES, ref_type)
        return []

    from concurrent.futures import ThreadPoolExecutor, as_completed
    ref_types = [rt for rt in ("person", "scene", "object")
                 if _items_for_type(gt_refs, rt) and _items_for_type(pred_refs, rt)]

    mapping: Dict[str, str] = {}
    if not ref_types:
        return mapping

    with ThreadPoolExecutor(max_workers=len(ref_types)) as pool:
        futures = {pool.submit(_match_one_type, rt): rt for rt in ref_types}
        for future in as_completed(futures):
            for pred_id, gt_id in future.result():
                mapping[pred_id] = gt_id

    return mapping


def greedy_match_by_score(candidates: List[Tuple[float, int, int]]) -> List[Tuple[int, int, float]]:
    candidates.sort(reverse=True)
    used_pred: Set[int] = set()
    used_gt: Set[int] = set()
    matches: List[Tuple[int, int, float]] = []
    for score, pi, gi in candidates:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi, score))
    return matches


def subshots_for(shot: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for idx, sub in enumerate(as_list(shot.get("visual_description"))):
        if isinstance(sub, dict):
            item = dict(sub)
            item["_sub_id"] = item.get("sub_shot_id") or f"{shot.get('shot_id', 'shot')}_sub_{idx}"
            out.append(item)
    return out


def subshot_refs(sub: Dict[str, Any]) -> Set[str]:
    return set(find_ref_ids(sub.get("sub_shot_content", "")))


def subshot_ref_time_ranges(shots: List[dict], ref_ids: Set[str]) -> List[List[float]]:
    target = {str(r).upper() for r in ref_ids if r}
    ranges: List[List[float]] = []
    if not target:
        return ranges
    for shot in shots:
        for sub in subshots_for(shot):
            if not (subshot_refs(sub) & target):
                continue
            sub_range = valid_range(sub.get("sub_shot_time"))
            if sub_range:
                ranges.append(sub_range)
    return ranges


def dialogue_lines(events: List[dict]) -> List[str]:
    dialogue_events = [
        event
        for event in events
        if str(event.get("type", "")).lower() == "dialogue"
    ]
    dialogue_events.sort(key=lambda e: (valid_range(e.get("time_range")) or [float("inf"), float("inf")])[0])
    return [line for line in (event_line(event).strip() for event in dialogue_events) if line]


def matched_subshot_groups(
    gt_shots: List[dict],
    pred_shots: List[dict],
    shot_matches: List[Tuple[int, int, float]],
    ref_mapping: Dict[str, str],
) -> Dict[str, Any]:
    groups = []
    used_pred: Set[Tuple[int, str]] = set()
    total_gt = sum(len(subshots_for(s)) for s in gt_shots)
    total_pred = sum(len(subshots_for(s)) for s in pred_shots)
    tious: List[float] = []
    who_scores: List[float] = []

    for pi, gi, _ in shot_matches:
        gt_shot = gt_shots[gi]
        pred_shot = pred_shots[pi]
        pred_subs = subshots_for(pred_shot)
        for gt_sub in subshots_for(gt_shot):
            gt_range = valid_range(gt_sub.get("sub_shot_time"))
            if not gt_range:
                continue
            window = [gt_range[0] - SUBSHOT_WINDOW_DELTA, gt_range[1] + SUBSHOT_WINDOW_DELTA]
            pred_group = []
            for pred_sub in pred_subs:
                if temporal_iou(pred_sub.get("sub_shot_time"), window) <= 0:
                    continue
                pred_group.append(pred_sub)
                used_pred.add((pi, str(pred_sub.get("_sub_id"))))
            if not pred_group:
                continue

            pred_ranges = [p.get("sub_shot_time") for p in pred_group]
            tious.append(range_to_union_iou(gt_sub.get("sub_shot_time"), pred_ranges))
            gt_refs = subshot_refs(gt_sub)
            pred_refs = map_refs((r for p in pred_group for r in subshot_refs(p)), ref_mapping)
            who_scores.append(1.0 if gt_refs.issubset(pred_refs) else 0.0 if gt_refs else 1.0)
            groups.append({
                "gt_shot_id": gt_shot.get("shot_id", ""),
                "pred_shot_id": pred_shot.get("shot_id", ""),
                "gt_subshot": gt_sub,
                "pred_subshots": pred_group,
            })

    matched_gt = len(groups)
    recall = matched_gt / (total_gt + EPS) if total_gt else None
    precision = len(used_pred) / (total_pred + EPS) if total_pred else (None if total_gt == 0 else 0.0)
    return {
        "groups": groups,
        "Subshot_Recall": recall,
        "Subshot_Precision": precision,
        "WHEN_S": safe_mean(tious) if tious else None,
        "WHO_sub": safe_mean(who_scores) if who_scores else None,
        "num_gt_subshots": total_gt,
        "num_pred_subshots": total_pred,
        "num_matched_gt_subshots": matched_gt,
        "num_matched_pred_subshots": len(used_pred),
    }


def match_subshots(
    gt_norm: List[dict],
    pred_norm: List[dict],
    shot_matches: List[Tuple[int, int, float]],
    ref_mapping: Dict[str, str],
) -> Dict[str, Any]:
    return matched_subshot_groups(gt_norm, pred_norm, shot_matches, ref_mapping)


def map_refs(refs: Iterable[str], ref_mapping: Dict[str, str]) -> Set[str]:
    return {ref_mapping.get(str(r).upper(), str(r).upper()) for r in refs if r}


def compute_sgc(gt: Dict[str, List[dict]], pred: Dict[str, List[dict]], schema_valid: float) -> Dict[str, float]:
    gt_refs, gt_events, gt_shots = gt["references"], gt["events"], gt["shots"]
    pred_refs, pred_events, pred_shots = pred["references"], pred["events"], pred["shots"]

    def record_score(records: List[dict], checks) -> float:
        if not records:
            return 0.0
        return safe_mean([safe_mean([1.0 if check(r) else 0.0 for check in checks]) for r in records])

    ref_complete = record_score(pred_refs, [
        lambda r: bool(r.get("ref_id")),
        lambda r: bool(r.get("type")),
        lambda r: bool(str(r.get("semantic_description", "")).strip()),
        lambda r: bool(detail_description(r).strip()),
    ])
    event_complete = record_score(pred_events, [
        lambda e: bool(e.get("event_id")),
        lambda e: bool(e.get("type")),
        lambda e: valid_range(e.get("time_range")) is not None,
        lambda e: bool(str(as_dict(e.get("content")).get("description", "")).strip()),
        lambda e: str(e.get("type", "")).lower() != "dialogue" or bool(str(as_dict(e.get("content")).get("line", "")).strip()),
    ])
    shot_complete = record_score(pred_shots, [
        lambda s: bool(s.get("shot_id")),
        lambda s: valid_range(s.get("time_range")) is not None,
        lambda s: "shot_transfer" in s,
        lambda s: isinstance(s.get("visual_description"), list),
        lambda s: isinstance(s.get("camera"), dict),
    ])
    sub_complete = safe_mean([
        record_score(subshots_for(s), [
            lambda x: valid_range(x.get("sub_shot_time")) is not None,
            lambda x: bool(str(x.get("sub_shot_content", "")).strip()),
        ])
        for s in pred_shots
    ])
    count_coverage = safe_mean([
        min(len(pred_refs) / max(len(gt_refs), 1), 1.0),
        min(len(pred_events) / max(len(gt_events), 1), 1.0),
        min(len(pred_shots) / max(len(gt_shots), 1), 1.0),
    ])
    link_complete = record_score(pred_shots, [
        lambda s: "active_events" in s and isinstance(s.get("active_events"), list),
        lambda s: "references_in_shot" in s and isinstance(s.get("references_in_shot"), list),
    ])
    sgc = safe_mean([schema_valid, ref_complete, event_complete, shot_complete, sub_complete, link_complete, count_coverage])
    return {
        "sgc_score": sgc,
        "sgc_schema_valid": schema_valid,
        "sgc_ref_completeness": ref_complete,
        "sgc_event_completeness": event_complete,
        "sgc_shot_completeness": shot_complete,
        "sgc_subshot_completeness": sub_complete,
        "sgc_link_field_completeness": link_complete,
        "sgc_count_coverage": count_coverage,
    }


def compute_evsa(gt: Dict[str, List[dict]], pred: Dict[str, List[dict]], event_matches, shot_matches) -> Dict[str, float]:
    pred_event_to_gt = {str(pred["events"][pi].get("event_id", "")).upper(): str(gt["events"][gi].get("event_id", "")).upper() for pi, gi, _ in event_matches}
    pred_shot_to_gt = {str(pred["shots"][pi].get("shot_id", "")).upper(): str(gt["shots"][gi].get("shot_id", "")).upper() for pi, gi, _ in shot_matches}
    gt_edges = {
        (str(eid).upper(), str(shot.get("shot_id", "")).upper())
        for shot in gt["shots"]
        for eid in as_list(shot.get("active_events"))
    }
    pred_edges = set()
    for shot in pred["shots"]:
        gt_sid = pred_shot_to_gt.get(str(shot.get("shot_id", "")).upper())
        if not gt_sid:
            continue
        for eid in as_list(shot.get("active_events")):
            gt_eid = pred_event_to_gt.get(str(eid).upper())
            if gt_eid:
                pred_edges.add((gt_eid, gt_sid))
    correct = len(pred_edges & gt_edges)
    precision = correct / (len(pred_edges) + EPS) if pred_edges else (None if not gt_edges else 0.0)
    recall = correct / (len(gt_edges) + EPS) if gt_edges else (None if not pred_edges else 0.0)
    evsa = f1_score(precision, recall) if precision is not None and recall is not None else None
    return {
        "evsa_precision": precision,
        "evsa_recall": recall,
        "evsa": evsa,
        "num_gt_event_shot_links": len(gt_edges),
        "num_pred_event_shot_links_mapped": len(pred_edges),
    }





def raw_shot_matches(gt_shots: List[dict], pred_shots: List[dict], threshold: float = SHOT_IOU_THRESHOLD) -> List[Tuple[int, int, float]]:
    candidates = []
    for pi, pred in enumerate(pred_shots):
        for gi, gt in enumerate(gt_shots):
            score = temporal_iou(pred.get("time_range"), gt.get("time_range"))
            if score >= threshold:
                candidates.append((score, pi, gi))
    return greedy_match_by_score(candidates)


def match_shots(gt_shots: List[dict], pred_shots: List[dict]) -> Dict[str, Any]:
    matches = raw_shot_matches(gt_shots, pred_shots)
    return {
        "matches": matches,
        "gt_for_match": gt_shots,
        "pred_for_match": pred_shots,
        "WHEN_H": safe_mean([s for _, _, s in matches]),
    }


def event_line(event: Dict[str, Any]) -> str:
    return str(as_dict(event.get("content")).get("line", "") or "")


def event_refs(event: Dict[str, Any]) -> Set[str]:
    content = as_dict(event.get("content"))
    refs = set(find_ref_ids(content))
    for key in ("speaker", "source_ref", "target_ref", "ref_id"):
        value = content.get(key, event.get(key))
        refs.update(find_ref_ids(value))
    return refs





def match_events(gt_events: List[dict], pred_events: List[dict]) -> Dict[str, Any]:
    candidates = []
    pair_scores: Dict[Tuple[int, int], Dict[str, float]] = {}
    for pi, pred in enumerate(pred_events):
        pred_type = str(pred.get("type", "")).lower()
        for gi, gt in enumerate(gt_events):
            gt_type = str(gt.get("type", "")).lower()
            if gt_type == "dialogue" and pred_type == "dialogue":
                score = wer_score(event_line(gt), event_line(pred))
                threshold = DIALOGUE_LINE_THRESHOLD
                line_score = score
            elif gt_type != "dialogue" and pred_type != "dialogue":
                score = temporal_iou(pred.get("time_range"), gt.get("time_range"))
                threshold = EVENT_TIME_THRESHOLD
                line_score = None
            else:
                continue
            time_score = temporal_iou(pred.get("time_range"), gt.get("time_range"))
            if score >= threshold:
                candidates.append((score, pi, gi))
                pair_scores[(pi, gi)] = {"match_score": score, "line_score": line_score, "time_score": time_score}
    matches = greedy_match_by_score(candidates)
    return {"matches": matches, "pair_scores": pair_scores}


def evaluate_one_video(gt_item: dict, pred_item: dict) -> Dict[str, Any]:
    gt, gt_diag = canonicalize_caption(gt_item.get("mtss_caption", {}))
    pred, pred_diag = canonicalize_caption(pred_item.get("model_prediction", {}))

    ref_mapping = match_references(gt["references"], pred["references"])
    shot_pack = match_shots(gt["shots"], pred["shots"])
    event_pack = match_events(gt["events"], pred["events"])
    event_matches = event_pack["matches"]
    shot_matches = shot_pack["matches"]
    sub = match_subshots(shot_pack["gt_for_match"], shot_pack["pred_for_match"], shot_matches, ref_mapping)

    matched_gt_refs = set(ref_mapping.values())
    ref_recall = len(matched_gt_refs) / (len(gt["references"]) + EPS) if gt["references"] else None
    ref_precision = len(ref_mapping) / (len(pred["references"]) + EPS) if pred["references"] else (None if not gt["references"] else 0.0)
    ref_compact = safe_mean([ref_recall, ref_precision])

    # Split Ref by type: Subject (person/animal/object) vs Scene
    def _ref_type(ref):
        return str(ref.get("type", "")).lower()

    gt_subject_refs = [r for r in gt["references"] if _ref_type(r) in ("person", "animal", "object")]
    gt_scene_refs = [r for r in gt["references"] if _ref_type(r) == "scene"]
    pred_subject_refs = [r for r in pred["references"] if _ref_type(r) in ("person", "animal", "object")]
    pred_scene_refs = [r for r in pred["references"] if _ref_type(r) == "scene"]

    gt_subject_ids = {str(r.get("ref_id", "")).upper() for r in gt_subject_refs}
    gt_scene_ids = {str(r.get("ref_id", "")).upper() for r in gt_scene_refs}
    pred_subject_ids = {str(r.get("ref_id", "")).upper() for r in pred_subject_refs}
    pred_scene_ids = {str(r.get("ref_id", "")).upper() for r in pred_scene_refs}

    matched_gt_subject = {gid for gid in matched_gt_refs if gid in gt_subject_ids}
    matched_gt_scene = {gid for gid in matched_gt_refs if gid in gt_scene_ids}
    matched_pred_subject = {pid for pid, gid in ref_mapping.items() if str(pid).upper() in pred_subject_ids}
    matched_pred_scene = {pid for pid, gid in ref_mapping.items() if str(pid).upper() in pred_scene_ids}

    ref_subject_recall = len(matched_gt_subject) / (len(gt_subject_refs) + EPS) if gt_subject_refs else None
    ref_subject_precision = len(matched_pred_subject) / (len(pred_subject_refs) + EPS) if pred_subject_refs else (None if not gt_subject_refs else 0.0)
    ref_subject_f1 = f1_score(ref_subject_precision, ref_subject_recall) if ref_subject_precision is not None and ref_subject_recall is not None else None

    ref_scene_recall = len(matched_gt_scene) / (len(gt_scene_refs) + EPS) if gt_scene_refs else None
    ref_scene_precision = len(matched_pred_scene) / (len(pred_scene_refs) + EPS) if pred_scene_refs else (None if not gt_scene_refs else 0.0)
    ref_scene_f1 = f1_score(ref_scene_precision, ref_scene_recall) if ref_scene_precision is not None and ref_scene_recall is not None else None

    event_ref_jaccards, dialogue_scores, event_time_ious = [], [], []
    speaker_correct, speaker_total = 0, 0
    for pi, gi, _ in event_matches:
        pred_ev, gt_ev = pred["events"][pi], gt["events"][gi]
        mapped_refs = map_refs(event_refs(pred_ev), ref_mapping)
        gt_refs = event_refs(gt_ev)
        event_ref_jaccards.append(len(mapped_refs & gt_refs) / (len(mapped_refs | gt_refs) + EPS) if (mapped_refs or gt_refs) else 1.0)
        if str(gt_ev.get("type", "")).lower() == "dialogue":
            line_score = event_pack["pair_scores"].get((pi, gi), {}).get("line_score")
            if line_score is not None:
                dialogue_scores.append(line_score)
            # Speaker F1: check if predicted speaker maps to correct GT speaker
            gt_content = as_dict(gt_ev.get("content"))
            pred_content = as_dict(pred_ev.get("content"))
            gt_speaker = str(gt_content.get("speaker", "")).upper().strip()
            pred_speaker = str(pred_content.get("speaker", "")).upper().strip()
            if gt_speaker:
                speaker_total += 1
                # Map predicted speaker through ref_mapping
                mapped_pred_speaker = ref_mapping.get(pred_speaker, pred_speaker)
                if mapped_pred_speaker == gt_speaker:
                    speaker_correct += 1
        event_time_ious.append(temporal_iou(pred_ev.get("time_range"), gt_ev.get("time_range")))

    speaker_f1_score = speaker_correct / (speaker_total + EPS) if speaker_total > 0 else None

    shot_ref_jaccards = []
    for pi, gi, _ in shot_matches:
        pred_shot, gt_shot = pred["shots"][pi], gt["shots"][gi]
        mapped = map_refs(as_list(pred_shot.get("references_in_shot")), ref_mapping)
        gt_refs = {str(x).upper() for x in as_list(gt_shot.get("references_in_shot"))}
        shot_ref_jaccards.append(len(mapped & gt_refs) / (len(mapped | gt_refs) + EPS) if (mapped or gt_refs) else 1.0)

    event_recall = len(event_matches) / (len(gt["events"]) + EPS) if gt["events"] else None
    event_precision = len(event_matches) / (len(pred["events"]) + EPS) if pred["events"] else (None if not gt["events"] else 0.0)
    event_f1 = f1_score(event_precision, event_recall) if event_precision is not None and event_recall is not None else None
    event_compact = event_f1
    shot_recall = len(shot_matches) / (len(gt["shots"]) + EPS) if gt["shots"] else None
    shot_precision = len(shot_matches) / (len(pred["shots"]) + EPS) if pred["shots"] else (None if not gt["shots"] else 0.0)
    shot_f1 = f1_score(shot_precision, shot_recall) if shot_precision is not None and shot_recall is not None else None

    evsa = compute_evsa(gt, pred, event_matches, shot_matches)
    when_h = shot_pack["WHEN_H"] if shot_matches else None
    when_e = safe_mean(event_time_ious) if event_matches else None
    when_s = sub["WHEN_S"]
    when = safe_mean([when_h, when_e, when_s])

    ccc_scores = []
    pred_ref_to_gt_ref = ref_mapping
    for gt_ref in gt["references"]:
        rid = str(gt_ref.get("ref_id", "")).upper()
        pred_ids = [pid for pid, gid in pred_ref_to_gt_ref.items() if gid == rid]
        gt_ranges = subshot_ref_time_ranges(gt["shots"], {rid})
        pred_ranges = subshot_ref_time_ranges(pred["shots"], set(pred_ids))
        ccc_scores.append(ranges_union_iou(gt_ranges, pred_ranges))

    dialogue_line = safe_mean(dialogue_scores) if dialogue_scores else None
    gt_dialogue_lines = dialogue_lines(gt["events"])
    pred_dialogue_lines = dialogue_lines(pred["events"])
    dialogue_merged = (
        wer_score(" ".join(gt_dialogue_lines), " ".join(pred_dialogue_lines))
        if gt_dialogue_lines or pred_dialogue_lines
        else None
    )
    subshot_f1 = (
        f1_score(sub["Subshot_Precision"], sub["Subshot_Recall"])
        if sub["Subshot_Precision"] is not None and sub["Subshot_Recall"] is not None
        else None
    )
    subshot_compact = subshot_f1
    refuse_compact = safe_mean([safe_mean(shot_ref_jaccards) if shot_ref_jaccards else None,
                                 safe_mean(event_ref_jaccards) if event_ref_jaccards else None,
                                 sub["WHO_sub"]])
    sgc = compute_sgc(gt, pred, pred_diag["schema_valid"])

    return {
        "video_path": gt_item.get("video_path", ""),
        "duration_sec": gt_item.get("duration_sec", 0),
        "duration_bucket": gt_item.get("duration_bucket", ""),
        "metric_version": "mtss_metrics_md_v1",
        "num_gt_refs": len(gt["references"]),
        "num_pred_refs": len(pred["references"]),
        "num_matched_refs": len(ref_mapping),
        "num_gt_events": len(gt["events"]),
        "num_pred_events": len(pred["events"]),
        "num_matched_events": len(event_matches),
        "num_gt_shots": len(gt["shots"]),
        "num_pred_shots": len(pred["shots"]),
        "num_matched_shots": len(shot_matches),
        "SGC": sgc["sgc_score"],
        "Ref": ref_compact,
        "RefUse": refuse_compact,
        "CCC": safe_mean(ccc_scores),
        "WHEN": when,
        "EVSA": evsa["evsa"],
        "Event": event_compact,
        "Dialog": dialogue_line,
        "Subshot": subshot_compact,
        "ref_recall": ref_recall,
        "ref_precision": ref_precision,
        "WHO_shot": safe_mean(shot_ref_jaccards) if shot_ref_jaccards else None,
        "WHO_event": safe_mean(event_ref_jaccards) if event_ref_jaccards else None,
        "WHO_sub": sub["WHO_sub"],
        "CCC_macro": safe_mean(ccc_scores),
        "WHEN_H": when_h,
        "WHEN_E": when_e,
        "WHEN_S": when_s,
        "shot_recall": shot_recall,
        "shot_precision": shot_precision,
        "shot_f1": shot_f1,
        "EVSA_P": evsa["evsa_precision"],
        "EVSA_R": evsa["evsa_recall"],
        "event_recall": event_recall,
        "event_precision": event_precision,
        "event_f1": event_f1,
        "DialogueLineScore": dialogue_line,
        "WHAT_dialogue_sem": dialogue_line,
        "DialogueMergedScore": dialogue_merged,
        "Subshot_Recall": sub["Subshot_Recall"],
        "Subshot_Precision": sub["Subshot_Precision"],
        "Subshot_F1": subshot_f1,
        # Paper-aligned: Ref Subject F1 / Ref Scene F1
        "Ref_Subject_F1": ref_subject_f1,
        "Ref_Subject_Recall": ref_subject_recall,
        "Ref_Subject_Precision": ref_subject_precision,
        "Ref_Scene_F1": ref_scene_f1,
        "Ref_Scene_Recall": ref_scene_recall,
        "Ref_Scene_Precision": ref_scene_precision,
        # Paper-aligned: Speaker F1
        "Speaker_F1": speaker_f1_score,
        # Paper-aligned: tIoU aliases
        "Shot_tIoU": when_h,
        "Event_tIoU": when_e,
        "Subshot_tIoU": when_s,
        "gt_schema_valid": gt_diag["schema_valid"],
        **{k: v for k, v in sgc.items()},
        **{k: v for k, v in sub.items() if k != "groups"},
        "num_gt_event_shot_links": evsa["num_gt_event_shot_links"],
        "num_pred_event_shot_links_mapped": evsa["num_pred_event_shot_links_mapped"],
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
                logger.warning("Skipping invalid JSON at %s:%d: %s", path, line_no, exc)
    return rows


def aggregate(per_video: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    numeric_keys = sorted(set(REPORT_METRIC_KEYS) | {
        key
        for row in per_video
        for key, value in row.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    })
    out: Dict[str, Dict[str, float]] = {}
    for key in numeric_keys:
        vals = [float(row[key]) for row in per_video if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))]
        if vals:
            out[key] = {
                "count": len(vals),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        else:
            out[key] = {
                "count": 0,
                "mean": None,
                "std": None,
                "median": None,
                "min": None,
                "max": None,
            }
    return out


def evaluate_batch(gt_path: str, pred_path: str, output_path: Optional[str] = None,
                    workers: int = 1) -> Dict[str, Any]:
    gt_rows = load_jsonl(gt_path)
    pred_rows = load_jsonl(pred_path)
    pred_lookup = {
        row.get("video_path", ""): row
        for row in pred_rows
        if row.get("video_path") and row.get("status") == "success"
    }
    logger.info("GT: %d videos, Pred: %d successful predictions", len(gt_rows), len(pred_lookup))

    pairs = []
    skipped = 0
    for gt_item in gt_rows:
        video_path = gt_item.get("video_path", "")
        pred_item = pred_lookup.get(video_path)
        if not pred_item:
            skipped += 1
            continue
        pairs.append((gt_item, pred_item))

    def _eval_safe(pair):
        gt_item, pred_item = pair
        try:
            return evaluate_one_video(gt_item, pred_item), None
        except Exception as exc:
            return None, (gt_item.get("video_path", ""), exc)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable=None, *args, **kwargs):
            if iterable is None:
                class _NoOpProgress:
                    def __enter__(self):
                        return self

                    def __exit__(self, exc_type, exc, tb):
                        return False

                    def update(self, *_args, **_kwargs):
                        return None

                return _NoOpProgress()
            return iterable

    per_video = []
    errors = 0
    effective_workers = max(1, workers)
    if effective_workers == 1:
        for pair in tqdm(pairs, desc="Evaluating videos (ref match)", unit="video"):
            result, err = _eval_safe(pair)
            if err:
                errors += 1
                logger.warning("Error evaluating %s: %s", err[0], err[1])
            else:
                per_video.append(result)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        logger.info("Parallel video evaluation with %d workers", effective_workers)
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(_eval_safe, p): i for i, p in enumerate(pairs)}
            indexed_results = {}
            with tqdm(total=len(pairs), desc="Evaluating videos (ref match)", unit="video") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    result, err = future.result()
                    if err:
                        errors += 1
                        logger.warning("Error evaluating %s: %s", err[0], err[1])
                    else:
                        indexed_results[idx] = result
                    pbar.update(1)
            per_video = [indexed_results[i] for i in sorted(indexed_results)]

    if not per_video:
        return {"error": "No videos evaluated", "num_gt_total": len(gt_rows), "num_skipped": skipped, "num_errors": errors}

    aggregated = aggregate(per_video)
    per_bucket: Dict[str, Any] = {}
    by_bucket: Dict[str, List[dict]] = defaultdict(list)
    for row in per_video:
        by_bucket[row.get("duration_bucket", "unknown")].append(row)
    for bucket, rows in by_bucket.items():
        per_bucket[bucket] = {"count": len(rows), "aggregated": aggregate(rows)}

    summary = {
        "metric_version": "mtss_metrics_md_v1",
        "num_evaluated": len(per_video),
        "num_skipped": skipped,
        "num_errors": errors,
        "num_gt_total": len(gt_rows),
        "aggregated": aggregated,
        "per_bucket": per_bucket,
        "main_table": {
            "SGC": aggregated.get("SGC", {}).get("mean"),
            "Ref": aggregated.get("Ref", {}).get("mean"),
            "RefUse": aggregated.get("RefUse", {}).get("mean"),
            "CCC": aggregated.get("CCC", {}).get("mean"),
            "WHEN": aggregated.get("WHEN", {}).get("mean"),
            "EVSA": aggregated.get("EVSA", {}).get("mean"),
            "Event": aggregated.get("Event", {}).get("mean"),
            "Dialog": aggregated.get("Dialog", {}).get("mean"),
            "Subshot": aggregated.get("Subshot", {}).get("mean"),
        },
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        detail_path = output_path.replace(".json", "_detail.jsonl")
        with open(detail_path, "w", encoding="utf-8") as f:
            for row in per_video:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Summary saved to: %s", output_path)
        logger.info("Per-video details saved to: %s", detail_path)

    return summary


def metric(summary: Dict[str, Any], key: str) -> float:
    value = summary.get("aggregated", {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else 0.0


def metric_or_none(summary: Dict[str, Any], key: str) -> Optional[float]:
    value = summary.get("aggregated", {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def print_bar(name: str, value: Optional[float]) -> None:
    if value is None:
        print(f"    {name:<28s} N/A")
        return
    value = max(0.0, min(1.0, value))
    bar = "#" * int(value * 30) + "." * (30 - int(value * 30))
    print(f"    {name:<28s} {bar} {value:.4f}")


def print_metric_value(label: str, value: Optional[float]) -> None:
    if value is None:
        print(f"    {label:<28s} N/A")
    else:
        print(f"    {label:<28s} {value:.4f}")


def print_summary(summary: Dict[str, Any]) -> None:
    if "error" in summary:
        print(summary["error"])
        return
    print("\n" + "=" * 72)
    print("  MTSS Rule-Based 4W Evaluation")
    print("=" * 72)
    print(f"  Videos evaluated: {summary.get('num_evaluated', 0)}")
    print(f"  Videos skipped:   {summary.get('num_skipped', 0)}")
    print(f"  Eval errors:       {summary.get('num_errors', 0)}")
    print()
    print("  Main Table: SGC + 8 rule-based metric families")
    print("  " + "-" * 60)
    for label, key in [
        ("SGC gate", "SGC"),
        ("WHO Ref", "Ref"),
        ("WHO RefUse", "RefUse"),
        ("WHO CCC", "CCC"),
        ("WHEN", "WHEN"),
        ("WHERE EVSA", "EVSA"),
        ("WHAT Event", "Event"),
        ("WHAT Dialog", "Dialog"),
        ("WHAT Subshot", "Subshot"),
    ]:
        print_bar(label, metric_or_none(summary, key))

    print()
    print("  Appendix / Expanded Rule Metrics")
    print("  " + "-" * 60)
    for label, key in [
        ("ref_recall", "ref_recall"),
        ("ref_precision", "ref_precision"),
        ("WHO_shot", "WHO_shot"),
        ("WHO_event", "WHO_event"),
        ("WHO_sub", "WHO_sub"),
        ("CCC_macro", "CCC_macro"),
        ("WHEN_H", "WHEN_H"),
        ("WHEN_E", "WHEN_E"),
        ("WHEN_S", "WHEN_S"),
        ("shot_recall", "shot_recall"),
        ("shot_precision", "shot_precision"),
        ("shot_f1", "shot_f1"),
        ("EVSA_P", "EVSA_P"),
        ("EVSA_R", "EVSA_R"),
        ("event_recall", "event_recall"),
        ("event_precision", "event_precision"),
        ("event_f1", "event_f1"),
        ("DialogueLineScore", "DialogueLineScore"),
        ("DialogueMergedScore", "DialogueMergedScore"),
        ("Subshot_Recall", "Subshot_Recall"),
        ("Subshot_Precision", "Subshot_Precision"),
        ("Subshot_F1", "Subshot_F1"),
    ]:
        print_metric_value(label, metric_or_none(summary, key))
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="MTSS rule-based 4W metrics")
    parser.add_argument("--gt", required=True, help="Path to GT JSONL")
    parser.add_argument("--pred", required=True, help="Path to prediction results.jsonl")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--verbose", action="store_true", help="Reserved for compatibility")
    parser.add_argument("--llm-judge", action="store_true", help="Deprecated compatibility flag; use mtss_llm_judge.py")
    parser.add_argument("--judge-base-url", default="", help=argparse.SUPPRESS)
    parser.add_argument("--judge-model", default="", help=argparse.SUPPRESS)
    parser.add_argument("--judge-api-key", default="", help=argparse.SUPPRESS)
    parser.add_argument("--judge-workers", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--judge-limit", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel video evaluation workers (default: 1 = sequential)")
    args = parser.parse_args()

    output = args.output or os.path.join(os.path.dirname(args.pred), "metrics_summary.json")
    if args.judge_base_url:
        os.environ["REF_MATCH_BASE_URL"] = args.judge_base_url
    if args.judge_model:
        os.environ["REF_MATCH_MODEL"] = args.judge_model
    if args.judge_api_key:
        os.environ["REF_MATCH_API_KEY"] = args.judge_api_key
    if args.llm_judge:
        logger.warning("--llm-judge is no longer mixed into rule metrics; run mtss_llm_judge.py separately.")
    summary = evaluate_batch(args.gt, args.pred, output, workers=args.workers)
    print_summary(summary)


if __name__ == "__main__":
    main()
