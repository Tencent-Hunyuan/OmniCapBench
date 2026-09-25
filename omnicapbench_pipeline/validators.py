#!/usr/bin/env python3
"""
MTSS Pipeline - Structural Validators
Programmatic hard validation layer for all rounds.
These checks run BEFORE Stage2 model-based evaluation to catch
structural / format errors that models might miss.

Validators:
  1. shot_time consistency (shot round): shots count & time_range must match shot_time
  2. ref_id format: must follow TYPE_N pattern
  3. active_events cross-reference: event_ids must exist in events list
  4. references_in_shot cross-reference: ref_ids must exist in references list
  5. time_range bounds: must be within video duration, no negative values
  6. event time_range ordering: start < end
  7. shot coverage: shots should cover full video without large gaps
  8. English-only content check: all descriptions must be in English
"""

import re
import logging
from typing import List, Optional

logger = logging.getLogger("omnicapbench_pipeline")

# ======================================================================
# ref_id format patterns
# ======================================================================
_REF_ID_PATTERN = re.compile(
    r"^(PERSON|ANIMAL|OBJECT|SCENE)_\d+"
    r"(_STYLE_\d+|_VIEW_\d+)?$"
)

_EVENT_ID_PATTERN = re.compile(
    r"^(DIALOGUE|SFX|MUSIC)_\d+$"
)

_SHOT_ID_PATTERN = re.compile(
    r"^shot_\d+$"
)


def validate_references(references: list, video_duration: float = None) -> List[str]:
    """
    Validate references array structure and format.

    Returns:
        List of error strings (empty = all good).
    """
    errors = []
    if not isinstance(references, list):
        errors.append("references must be a list")
        return errors

    seen_ids = set()
    for i, ref in enumerate(references):
        if not isinstance(ref, dict):
            errors.append(f"references[{i}]: must be a dict")
            continue

        # ref_id format
        ref_id = ref.get("ref_id", "")
        if not ref_id:
            errors.append(f"references[{i}]: missing ref_id")
        elif not _REF_ID_PATTERN.match(ref_id):
            errors.append(
                f"references[{i}]: ref_id '{ref_id}' does not match "
                f"pattern TYPE_N (e.g., PERSON_1, OBJECT_1, SCENE_1_VIEW_1)"
            )
        if ref_id in seen_ids:
            errors.append(f"references[{i}]: duplicate ref_id '{ref_id}'")
        seen_ids.add(ref_id)

        # type field
        ref_type = ref.get("type", "")
        valid_types = {"person", "animal", "object", "scene"}
        if ref_type not in valid_types:
            errors.append(
                f"{ref_id}: type '{ref_type}' not in {valid_types}"
            )

        # type-specific checks
        if ref_type != "person":
            attrs = (ref.get("appearance_anchor", {})
                     .get("attributes", None))
            if attrs and attrs != {}:
                errors.append(
                    f"{ref_id}: type='{ref_type}' but attributes is not empty {{}}"
                )

        if ref_type == "scene":
            ts = (ref.get("appearance_anchor", {})
                  .get("id_features", {}).get("timestamp", None))
            if ts is not None and ts != "":
                errors.append(
                    f"{ref_id}: type='scene' but timestamp is not empty string"
                )

        # timestamp bounds check
        if ref_type != "scene" and video_duration:
            ts = (ref.get("appearance_anchor", {})
                  .get("id_features", {}).get("timestamp", ""))
            if ts:
                try:
                    ts_val = float(ts)
                    if ts_val < 0 or ts_val > video_duration + 1.0:
                        errors.append(
                            f"{ref_id}: timestamp {ts_val} out of video range [0, {video_duration}]"
                        )
                except (ValueError, TypeError):
                    pass  # non-numeric timestamp, model might use different format

    return errors


def validate_events(events: list, references: list = None,
                    video_duration: float = None,
                    allow_narrator_speaker: bool = True) -> List[str]:
    """
    Validate events array structure, format, and cross-references.

    Args:
        events: events array to validate
        references: references array for cross-reference checking
        video_duration: video duration for time_range bounds checking
        allow_narrator_speaker: if True, allow non-ref_id descriptive speakers
            (e.g., "narrator", "voiceover"). Only event round should set this
            to True; other rounds should set to False.

    Returns:
        List of error strings.
    """
    errors = []
    if not isinstance(events, list):
        errors.append("events must be a list")
        return errors

    # Build valid ref_ids set for cross-reference checking
    valid_ref_ids = set()
    if references:
        for ref in references:
            if isinstance(ref, dict):
                valid_ref_ids.add(ref.get("ref_id", ""))

    seen_ids = set()
    for i, evt in enumerate(events):
        if not isinstance(evt, dict):
            errors.append(f"events[{i}]: must be a dict")
            continue

        # event_id format
        event_id = evt.get("event_id", "")
        if not event_id:
            errors.append(f"events[{i}]: missing event_id")
        elif not _EVENT_ID_PATTERN.match(event_id):
            errors.append(
                f"events[{i}]: event_id '{event_id}' does not match "
                f"pattern TYPE_N (e.g., DIALOGUE_1, SFX_1, MUSIC_1)"
            )
        if event_id in seen_ids:
            errors.append(f"events[{i}]: duplicate event_id '{event_id}'")
        seen_ids.add(event_id)

        # type field
        evt_type = evt.get("type", "")
        valid_types = {"dialogue", "sfx", "music"}
        if evt_type not in valid_types:
            errors.append(f"{event_id}: type '{evt_type}' not in {valid_types}")

        # time_range validation
        tr = evt.get("time_range", [])
        if not isinstance(tr, list) or len(tr) != 2:
            errors.append(f"{event_id}: time_range must be [start, end]")
        else:
            start, end = tr
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                errors.append(f"{event_id}: time_range values must be numbers")
            else:
                if start < 0:
                    errors.append(f"{event_id}: time_range start {start} < 0")
                if end < start:
                    errors.append(f"{event_id}: time_range end {end} < start {start}")
                if video_duration and end > video_duration + 3.0:
                    errors.append(
                        f"{event_id}: time_range end {end} exceeds video duration {video_duration}"
                    )

        # speaker cross-reference
        # NOTE: speaker can be:
        #   1. A valid ref_id (e.g., PERSON_1) — must exist in references
        #   2. A descriptive string for narrators/voiceovers (e.g., "narrator",
        #      "male voiceover", "off-screen narrator") — ONLY acceptable in event round
        content = evt.get("content", {})
        if isinstance(content, dict):
            speaker = content.get("speaker", "")
            if speaker and valid_ref_ids and speaker not in valid_ref_ids:
                if _REF_ID_PATTERN.match(speaker):
                    # Speaker looks like a ref_id but doesn't exist in references
                    errors.append(
                        f"{event_id}: speaker '{speaker}' looks like a ref_id "
                        f"but is not found in references"
                    )
                elif allow_narrator_speaker:
                    # Non-ref_id speaker (e.g., "narrator", "voiceover")
                    # Only acceptable in event round
                    logger.debug(
                        f"[Validator] {event_id}: speaker '{speaker}' is not a ref_id, "
                        f"treated as narrator/voiceover (acceptable in event round)"
                    )
                else:
                    # Non-ref_id speaker in non-event round — not allowed
                    errors.append(
                        f"{event_id}: speaker '{speaker}' is not a valid ref_id. "
                        f"Descriptive speaker names (narrator/voiceover) are only "
                        f"allowed in the event round."
                    )

    return errors


def validate_shots(shots: list, shot_timestamps: list = None,
                   references: list = None, events: list = None,
                   video_duration: float = None) -> List[str]:
    """
    Validate shots array structure, format, cross-references,
    and shot_time consistency.

    Returns:
        List of error strings.
    """
    errors = []
    if not isinstance(shots, list):
        errors.append("shots must be a list")
        return errors

    # Build valid ref_ids and event_ids for cross-reference checking
    valid_ref_ids = set()
    if references:
        for ref in references:
            if isinstance(ref, dict):
                valid_ref_ids.add(ref.get("ref_id", ""))

    valid_event_ids = set()
    event_time_ranges = {}
    if events:
        for evt in events:
            if isinstance(evt, dict):
                eid = evt.get("event_id", "")
                valid_event_ids.add(eid)
                event_time_ranges[eid] = evt.get("time_range", [])

    # ================================================================
    # CRITICAL: shot_time consistency check
    # ================================================================
    if shot_timestamps:
        if len(shots) != len(shot_timestamps):
            errors.append(
                f"CRITICAL: shots count ({len(shots)}) does not match "
                f"shot_time count ({len(shot_timestamps)}). "
                f"Expected {len(shot_timestamps)} shots based on shot_time: {shot_timestamps}"
            )

        # Check each shot's time_range against shot_time
        for j, (shot, expected_tr) in enumerate(
            zip(shots, shot_timestamps[:len(shots)])
        ):
            if not isinstance(shot, dict):
                continue
            actual_tr = shot.get("time_range", [])
            if isinstance(actual_tr, list) and len(actual_tr) == 2:
                exp_start, exp_end = expected_tr
                act_start, act_end = actual_tr
                # Tolerance: ±0.5s for shot boundaries
                if abs(act_start - exp_start) > 0.5:
                    errors.append(
                        f"CRITICAL: shot_{j+1} start time {act_start} does not match "
                        f"shot_time start {exp_start} (tolerance ±0.5s)"
                    )
                if abs(act_end - exp_end) > 0.5:
                    errors.append(
                        f"CRITICAL: shot_{j+1} end time {act_end} does not match "
                        f"shot_time end {exp_end} (tolerance ±0.5s)"
                    )

    seen_ids = set()
    for i, shot in enumerate(shots):
        if not isinstance(shot, dict):
            errors.append(f"shots[{i}]: must be a dict")
            continue

        # shot_id format
        shot_id = shot.get("shot_id", "")
        if not shot_id:
            errors.append(f"shots[{i}]: missing shot_id")
        elif not _SHOT_ID_PATTERN.match(shot_id):
            errors.append(
                f"shots[{i}]: shot_id '{shot_id}' does not match "
                f"pattern shot_N (e.g., shot_1, shot_2)"
            )
        if shot_id in seen_ids:
            errors.append(f"shots[{i}]: duplicate shot_id '{shot_id}'")
        seen_ids.add(shot_id)

        # time_range validation
        tr = shot.get("time_range", [])
        if not isinstance(tr, list) or len(tr) != 2:
            errors.append(f"{shot_id}: time_range must be [start, end]")
        else:
            start, end = tr
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                errors.append(f"{shot_id}: time_range values must be numbers")
            else:
                if start < 0:
                    errors.append(f"{shot_id}: time_range start {start} < 0")
                if end <= start:
                    errors.append(f"{shot_id}: time_range end {end} <= start {start}")
                if video_duration and end > video_duration + 3.0:
                    errors.append(
                        f"{shot_id}: time_range end {end} exceeds video duration {video_duration}"
                    )

        # visual_description validation
        vd = shot.get("visual_description", [])
        if not isinstance(vd, list):
            errors.append(f"{shot_id}: visual_description must be a list")
        elif len(vd) == 0:
            errors.append(f"{shot_id}: visual_description is empty")
        else:
            for k, sub in enumerate(vd):
                if not isinstance(sub, dict):
                    errors.append(f"{shot_id}: visual_description[{k}] must be a dict")
                    continue
                sst = sub.get("sub_shot_time", [])
                if not isinstance(sst, list) or len(sst) != 2:
                    errors.append(
                        f"{shot_id}: visual_description[{k}].sub_shot_time "
                        f"must be [start, end]"
                    )
                ssc = sub.get("sub_shot_content", "")
                if not ssc:
                    errors.append(
                        f"{shot_id}: visual_description[{k}].sub_shot_content is empty"
                    )

        # active_events cross-reference
        active = shot.get("active_events", [])
        if isinstance(active, list) and valid_event_ids:
            for ae in active:
                if ae not in valid_event_ids:
                    errors.append(
                        f"{shot_id}: active_events references '{ae}' "
                        f"which does not exist in events"
                    )
                # Check time overlap
                if ae in event_time_ranges and isinstance(tr, list) and len(tr) == 2:
                    etr = event_time_ranges[ae]
                    if isinstance(etr, list) and len(etr) == 2:
                        # Events and shots should have time overlap
                        # Use generous tolerance of 3s
                        if etr[1] + 3.0 < tr[0] or etr[0] - 3.0 > tr[1]:
                            errors.append(
                                f"{shot_id}: active_events '{ae}' time_range "
                                f"{etr} has no overlap with shot time_range {tr}"
                            )

        # references_in_shot cross-reference
        refs_in = shot.get("references_in_shot", [])
        if isinstance(refs_in, list) and valid_ref_ids:
            for rid in refs_in:
                if rid not in valid_ref_ids:
                    errors.append(
                        f"{shot_id}: references_in_shot '{rid}' "
                        f"not found in references"
                    )

        # camera field check
        camera = shot.get("camera", {})
        if not isinstance(camera, dict):
            errors.append(f"{shot_id}: camera must be a dict")
        else:
            for field in ["shot_type", "angle", "movement"]:
                if field not in camera:
                    errors.append(f"{shot_id}: camera missing '{field}' field")

        # Consistency check: ref_ids used in visual_description vs references_in_shot
        # If a [ref_id] is used in sub_shot_content, it should be in references_in_shot
        refs_in = shot.get("references_in_shot", [])
        refs_in_set = set(refs_in) if isinstance(refs_in, list) else set()
        vd = shot.get("visual_description", [])
        if isinstance(vd, list):
            for k, sub in enumerate(vd):
                if not isinstance(sub, dict):
                    continue
                ssc = sub.get("sub_shot_content", "")
                if ssc:
                    # Extract [REF_ID] patterns from sub_shot_content
                    used_refs = re.findall(r'\[([A-Z]+_\d+(?:_STYLE_\d+|_VIEW_\d+)?)\]', ssc)
                    for used_ref in used_refs:
                        if used_ref not in refs_in_set:
                            if valid_ref_ids and used_ref in valid_ref_ids:
                                errors.append(
                                    f"{shot_id}: visual_description[{k}] uses [{used_ref}] "
                                    f"but it is not listed in references_in_shot"
                                )
                            elif valid_ref_ids and used_ref not in valid_ref_ids:
                                errors.append(
                                    f"{shot_id}: visual_description[{k}] uses [{used_ref}] "
                                    f"which does not exist in references"
                                )

    # Check shot coverage (no large gaps)
    if len(shots) >= 2:
        for i in range(len(shots) - 1):
            tr_curr = shots[i].get("time_range", [])
            tr_next = shots[i + 1].get("time_range", [])
            if (isinstance(tr_curr, list) and len(tr_curr) == 2 and
                isinstance(tr_next, list) and len(tr_next) == 2):
                gap = tr_next[0] - tr_curr[1]
                if gap > 1.0:
                    errors.append(
                        f"Gap of {gap:.1f}s between shot_{i+1} end ({tr_curr[1]}) "
                        f"and shot_{i+2} start ({tr_next[0]})"
                    )
                overlap = tr_curr[1] - tr_next[0]
                if overlap > 1.0:
                    errors.append(
                        f"Overlap of {overlap:.1f}s between shot_{i+1} end ({tr_curr[1]}) "
                        f"and shot_{i+2} start ({tr_next[0]})"
                    )

    return errors


def validate_stage1_output(
    round_name: str,
    generated: dict,
    shot_timestamps: list = None,
    references: list = None,
    events: list = None,
    video_duration: float = None,
) -> List[str]:
    """
    Master validation function for Stage1 output.
    Runs the appropriate validator based on round_name.

    Returns:
        List of error strings (empty = all good).
    """
    errors = []

    if round_name == "reference_bank":
        refs = generated.get("references", [])
        errors = validate_references(refs, video_duration)

    elif round_name == "event":
        evts = generated.get("events", [])
        # Event round: allow narrator/voiceover descriptive speakers
        errors = validate_events(evts, references, video_duration,
                                 allow_narrator_speaker=True)

    elif round_name == "shot":
        shots = generated.get("shots", [])
        errors = validate_shots(
            shots, shot_timestamps, references, events, video_duration
        )

    if errors:
        logger.warning(
            f"[Validator] {round_name}: {len(errors)} structural issues found"
        )
        for e in errors:
            logger.warning(f"[Validator]   - {e}")

    return errors


# ======================================================================
# Auto-fix: attempt to programmatically fix minor structural issues
# before validation, to avoid unnecessary Stage1 retries.
# ======================================================================

def auto_fix_structural_issues(
    round_name: str,
    generated: dict,
    shot_timestamps: list = None,
    references: list = None,
    events: list = None,
    video_duration: float = None,
) -> List[str]:
    """
    Attempt to auto-fix minor structural issues in Stage1 output.
    Modifies `generated` dict IN-PLACE and returns a list of fix descriptions.

    Only fixes issues that are unambiguous and safe to correct programmatically.
    Does NOT touch any semantic/descriptive content — only structural fields.

    Returns:
        List of human-readable fix descriptions (empty = nothing fixed).
    """
    fixes = []

    if round_name == "reference_bank":
        fixes.extend(_fix_references(generated.get("references", []),
                                     video_duration))
    elif round_name == "event":
        fixes.extend(_fix_events(generated.get("events", []),
                                 references, video_duration))
    elif round_name == "shot":
        fixes.extend(_fix_shots(generated.get("shots", []),
                                shot_timestamps, references, events,
                                video_duration))

    if fixes:
        logger.info(f"[AutoFix] {round_name}: applied {len(fixes)} auto-fixes")
        for f in fixes:
            logger.info(f"[AutoFix]   ✔ {f}")

    return fixes


def _fix_references(references: list, video_duration: float = None) -> List[str]:
    """Auto-fix reference structural issues."""
    fixes = []
    if not isinstance(references, list):
        return fixes

    for i, ref in enumerate(references):
        if not isinstance(ref, dict):
            continue

        ref_type = ref.get("type", "")

        # Fix: scene timestamp should be ""
        if ref_type == "scene":
            ts = (ref.get("appearance_anchor", {})
                  .get("id_features", {}).get("timestamp", None))
            if ts is not None and ts != "":
                ref.setdefault("appearance_anchor", {}).setdefault(
                    "id_features", {})["timestamp"] = ""
                fixes.append(
                    f"references[{i}] ({ref.get('ref_id', '?')}): "
                    f"type='scene', set timestamp to '' (was '{ts}')"
                )

        # Fix: non-person attributes should be {}
        if ref_type not in ("person",) and ref_type:
            attrs = (ref.get("appearance_anchor", {})
                     .get("attributes", None))
            if attrs and attrs != {}:
                ref.setdefault("appearance_anchor", {})["attributes"] = {}
                fixes.append(
                    f"references[{i}] ({ref.get('ref_id', '?')}): "
                    f"type='{ref_type}', cleared non-empty attributes to {{}}"
                )

        # Fix: timestamp out of video range → clamp
        if ref_type != "scene" and video_duration:
            ts_str = (ref.get("appearance_anchor", {})
                      .get("id_features", {}).get("timestamp", ""))
            if ts_str:
                try:
                    ts_val = float(ts_str)
                    if ts_val < 0:
                        ref["appearance_anchor"]["id_features"]["timestamp"] = "0.0"
                        fixes.append(
                            f"references[{i}] ({ref.get('ref_id', '?')}): "
                            f"timestamp {ts_val} < 0, clamped to 0.0"
                        )
                    elif ts_val > video_duration:
                        new_ts = str(round(video_duration, 1))
                        ref["appearance_anchor"]["id_features"]["timestamp"] = new_ts
                        fixes.append(
                            f"references[{i}] ({ref.get('ref_id', '?')}): "
                            f"timestamp {ts_val} > video duration {video_duration}, "
                            f"clamped to {new_ts}"
                        )
                except (ValueError, TypeError):
                    pass

    return fixes


def _fix_events(events: list, references: list = None,
                video_duration: float = None) -> List[str]:
    """Auto-fix event structural issues."""
    fixes = []
    if not isinstance(events, list):
        return fixes

    for i, evt in enumerate(events):
        if not isinstance(evt, dict):
            continue

        event_id = evt.get("event_id", f"events[{i}]")

        # Fix: time_range start < 0 → clamp to 0
        tr = evt.get("time_range", [])
        if isinstance(tr, list) and len(tr) == 2:
            start, end = tr
            changed = False
            if isinstance(start, (int, float)) and start < 0:
                tr[0] = 0.0
                changed = True
                fixes.append(f"{event_id}: time_range start {start} < 0, clamped to 0.0")
            if isinstance(end, (int, float)) and video_duration and end > video_duration + 1.0:
                tr[1] = round(video_duration, 2)
                changed = True
                fixes.append(
                    f"{event_id}: time_range end {end} > video duration "
                    f"{video_duration}, clamped to {tr[1]}"
                )
            if changed:
                evt["time_range"] = tr

    return fixes


def _fix_shots(shots: list, shot_timestamps: list = None,
               references: list = None, events: list = None,
               video_duration: float = None) -> List[str]:
    """Auto-fix shot structural issues."""
    fixes = []
    if not isinstance(shots, list):
        return fixes

    # Build valid ref_ids and event_ids
    valid_ref_ids = set()
    if references:
        for ref in references:
            if isinstance(ref, dict):
                valid_ref_ids.add(ref.get("ref_id", ""))

    valid_event_ids = set()
    if events:
        for evt in events:
            if isinstance(evt, dict):
                valid_event_ids.add(evt.get("event_id", ""))

    # Fix: shot_id format normalization (e.g., "Shot_1" → "shot_1", "shot1" → "shot_1")
    for i, shot in enumerate(shots):
        if not isinstance(shot, dict):
            continue

        shot_id = shot.get("shot_id", "")
        if shot_id and not _SHOT_ID_PATTERN.match(shot_id):
            # Try to normalize: "Shot_1" → "shot_1", "shot1" → "shot_1"
            normalized = shot_id.lower().strip()
            # "shot1" → "shot_1"
            m = re.match(r'^shot[_\s-]?(\d+)$', normalized)
            if m:
                new_id = f"shot_{m.group(1)}"
                shot["shot_id"] = new_id
                fixes.append(f"shots[{i}]: shot_id '{shot_id}' → '{new_id}'")
            else:
                # Fallback: assign sequential shot_id
                new_id = f"shot_{i + 1}"
                shot["shot_id"] = new_id
                fixes.append(
                    f"shots[{i}]: invalid shot_id '{shot_id}' → '{new_id}'"
                )

    # Fix: shot time_range alignment to shot_timestamps (within tolerance)
    if shot_timestamps and len(shots) == len(shot_timestamps):
        for j, (shot, expected_tr) in enumerate(zip(shots, shot_timestamps)):
            if not isinstance(shot, dict):
                continue
            actual_tr = shot.get("time_range", [])
            if not isinstance(actual_tr, list) or len(actual_tr) != 2:
                continue

            exp_start, exp_end = expected_tr
            act_start, act_end = actual_tr
            changed = False

            # Auto-align if within reasonable tolerance (≤3s)
            # This avoids CRITICAL errors for minor time misalignment
            if isinstance(act_start, (int, float)) and isinstance(exp_start, (int, float)):
                diff_start = abs(act_start - exp_start)
                if 0 < diff_start <= 3.0:
                    actual_tr[0] = exp_start
                    changed = True

            if isinstance(act_end, (int, float)) and isinstance(exp_end, (int, float)):
                diff_end = abs(act_end - exp_end)
                if 0 < diff_end <= 3.0:
                    actual_tr[1] = exp_end
                    changed = True

            if changed:
                shot["time_range"] = actual_tr
                fixes.append(
                    f"shot_{j+1}: time_range [{act_start}, {act_end}] → "
                    f"[{actual_tr[0]}, {actual_tr[1]}] "
                    f"(aligned to shot_time [{exp_start}, {exp_end}])"
                )

    # Per-shot fixes
    for i, shot in enumerate(shots):
        if not isinstance(shot, dict):
            continue

        shot_id = shot.get("shot_id", f"shots[{i}]")

        # Fix: time_range start < 0 → clamp to 0
        tr = shot.get("time_range", [])
        if isinstance(tr, list) and len(tr) == 2:
            start, end = tr
            if isinstance(start, (int, float)) and start < 0:
                tr[0] = 0.0
                shot["time_range"] = tr
                fixes.append(f"{shot_id}: time_range start {start} < 0, clamped to 0.0")
            if isinstance(end, (int, float)) and video_duration and end > video_duration + 1.0:
                tr[1] = round(video_duration, 2)
                shot["time_range"] = tr
                fixes.append(
                    f"{shot_id}: time_range end {end} > video duration "
                    f"{video_duration}, clamped to {tr[1]}"
                )

        # Fix: remove invalid active_events references
        active = shot.get("active_events", [])
        if isinstance(active, list) and valid_event_ids:
            invalid_events = [ae for ae in active if ae not in valid_event_ids]
            if invalid_events:
                shot["active_events"] = [ae for ae in active if ae in valid_event_ids]
                fixes.append(
                    f"{shot_id}: removed invalid active_events {invalid_events}"
                )

        # Fix: remove invalid references_in_shot references
        refs_in = shot.get("references_in_shot", [])
        if isinstance(refs_in, list) and valid_ref_ids:
            invalid_refs = [rid for rid in refs_in if rid not in valid_ref_ids]
            if invalid_refs:
                shot["references_in_shot"] = [rid for rid in refs_in if rid in valid_ref_ids]
                fixes.append(
                    f"{shot_id}: removed invalid references_in_shot {invalid_refs}"
                )

        # Fix: camera missing fields → add defaults
        camera = shot.get("camera", {})
        if isinstance(camera, dict):
            camera_defaults = {
                "shot_type": "medium shot",
                "angle": "eye level",
                "movement": "static",
            }
            for field, default_val in camera_defaults.items():
                if field not in camera:
                    camera[field] = default_val
                    fixes.append(
                        f"{shot_id}: camera missing '{field}', set to '{default_val}'"
                    )
            shot["camera"] = camera
        elif not isinstance(camera, dict):
            shot["camera"] = {
                "shot_type": "medium shot",
                "angle": "eye level",
                "movement": "static",
            }
            fixes.append(f"{shot_id}: camera was not a dict, set to defaults")

        # Fix: sub_shot_time clamping within shot time_range
        vd = shot.get("visual_description", [])
        if isinstance(vd, list) and isinstance(tr, list) and len(tr) == 2:
            shot_start = float(tr[0]) if isinstance(tr[0], (int, float)) else 0
            shot_end = float(tr[1]) if isinstance(tr[1], (int, float)) else 0
            for k, sub in enumerate(vd):
                if not isinstance(sub, dict):
                    continue
                sst = sub.get("sub_shot_time", [])
                if isinstance(sst, list) and len(sst) == 2:
                    s, e = sst
                    if isinstance(s, (int, float)) and isinstance(e, (int, float)):
                        new_s, new_e = s, e
                        clamped = False
                        if new_s < shot_start:
                            new_s = shot_start
                            clamped = True
                        if new_e > shot_end:
                            new_e = shot_end
                            clamped = True
                        if new_s > shot_end:
                            new_s = shot_start
                            clamped = True
                        if new_e < shot_start:
                            new_e = shot_end
                            clamped = True
                        if new_s >= new_e:
                            new_s = shot_start
                            new_e = shot_end
                            clamped = True
                        if clamped:
                            sub["sub_shot_time"] = [round(new_s, 2), round(new_e, 2)]
                            fixes.append(
                                f"{shot_id}.visual_description[{k}]: "
                                f"sub_shot_time [{s}, {e}] → "
                                f"[{round(new_s, 2)}, {round(new_e, 2)}]"
                            )
                elif not sst or not isinstance(sst, list) or len(sst) != 2:
                    # Missing sub_shot_time → set to shot range
                    if isinstance(tr, list) and len(tr) == 2:
                        sub["sub_shot_time"] = [float(tr[0]), float(tr[1])]
                        fixes.append(
                            f"{shot_id}.visual_description[{k}]: "
                            f"missing sub_shot_time, set to shot range "
                            f"[{tr[0]}, {tr[1]}]"
                        )

        # Fix: auto-add missing ref_ids to references_in_shot
        # If visual_description uses [REF_ID] that exists in references but is
        # not listed in references_in_shot, add it automatically.
        refs_in = shot.get("references_in_shot", [])
        if isinstance(refs_in, list):
            refs_in_set = set(refs_in)
            vd_list = shot.get("visual_description", [])
            added_refs = []
            if isinstance(vd_list, list):
                for sub in vd_list:
                    if not isinstance(sub, dict):
                        continue
                    ssc = sub.get("sub_shot_content", "")
                    if ssc:
                        used_refs = re.findall(
                            r'\[([A-Z]+_\d+(?:_STYLE_\d+|_VIEW_\d+)?)\]', ssc
                        )
                        for used_ref in used_refs:
                            if (used_ref not in refs_in_set
                                    and used_ref in valid_ref_ids):
                                refs_in.append(used_ref)
                                refs_in_set.add(used_ref)
                                added_refs.append(used_ref)
            if added_refs:
                shot["references_in_shot"] = refs_in
                fixes.append(
                    f"{shot_id}: auto-added {added_refs} to references_in_shot "
                    f"(used in visual_description but was missing)"
                )

    return fixes


def format_validation_feedback(errors: List[str]) -> str:
    """
    Format validation errors into a feedback string for Stage1 re-generation.
    Separates CRITICAL errors from other errors.
    """
    if not errors:
        return ""

    critical = [e for e in errors if e.startswith("CRITICAL")]
    others = [e for e in errors if not e.startswith("CRITICAL")]

    parts = []
    parts.append("=== Structural Validation Errors (Programmatic Check) ===")
    parts.append("The following structural issues were found in your output:")
    parts.append("")

    if critical:
        parts.append("** CRITICAL ERRORS (must fix) **")
        for e in critical:
            parts.append(f"  - {e}")
        parts.append("")

    if others:
        parts.append("** Other structural issues **")
        for e in others:
            parts.append(f"  - {e}")
        parts.append("")

    parts.append("Please fix ALL the above issues in your next generation.")
    parts.append("Pay special attention to CRITICAL errors — these indicate "
                 "fundamental structural problems that must be resolved.")

    return "\n".join(parts)


def get_video_duration(video_path: str) -> Optional[float]:
    """
    Get video duration in seconds using ffprobe.
    Returns None if unable to determine.
    """
    import subprocess
    from . import config

    try:
        cmd = [
            config.FFPROBE_PATH, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        duration = float(result.stdout.strip())
        return round(duration, 2)
    except Exception as e:
        logger.warning(f"[Validator] Failed to get video duration: {e}")
        return None


# ======================================================================
# Post-processing: fix sub_shot_time out-of-bounds (after Stage4 Select)
# ======================================================================

def fix_sub_shot_times(round_result: dict, shot_timestamps: list = None) -> dict:
    """
    Post-processing step after Stage4 Select for the shot round.
    Clamps all sub_shot_time values to be within their parent shot's time_range.

    This fixes cases where Stage4 Select might assemble sub_shots from
    different candidates that have slightly misaligned time boundaries.

    Rules:
      - sub_shot_time[0] (start) is clamped to [shot_start, shot_end]
      - sub_shot_time[1] (end) is clamped to [shot_start, shot_end]
      - If start >= end after clamping, set to [shot_start, shot_end] (full shot)
      - If sub_shot_time is missing or malformed, set to [shot_start, shot_end]

    Args:
        round_result: The shot round output dict containing "shots" key
        shot_timestamps: Optional list of [start, end] for each shot (for reference)

    Returns:
        The fixed round_result dict (modified in-place and returned)
    """
    shots = round_result.get("shots", [])
    if not shots:
        return round_result

    fix_count = 0
    for shot in shots:
        shot_range = shot.get("time_range")
        if not shot_range or len(shot_range) < 2:
            continue

        shot_start = float(shot_range[0])
        shot_end = float(shot_range[1])

        visual_desc = shot.get("visual_description", [])
        if not visual_desc:
            continue

        for sub in visual_desc:
            sub_time = sub.get("sub_shot_time")

            # Handle missing or malformed sub_shot_time
            if not sub_time or not isinstance(sub_time, (list, tuple)) or len(sub_time) < 2:
                sub["sub_shot_time"] = [shot_start, shot_end]
                fix_count += 1
                logger.debug(
                    f"[PostProcess] {shot.get('shot_id', '?')}: "
                    f"Missing sub_shot_time, set to shot range [{shot_start}, {shot_end}]"
                )
                continue

            orig_start = float(sub_time[0])
            orig_end = float(sub_time[1])
            new_start = orig_start
            new_end = orig_end
            needs_fix = False

            # Clamp start to [shot_start, shot_end]
            if new_start < shot_start:
                new_start = shot_start
                needs_fix = True
            if new_start > shot_end:
                new_start = shot_end
                needs_fix = True

            # Clamp end to [shot_start, shot_end]
            if new_end < shot_start:
                new_end = shot_start
                needs_fix = True
            if new_end > shot_end:
                new_end = shot_end
                needs_fix = True

            # Ensure start < end (with small tolerance)
            if new_start >= new_end:
                # If clamping made them equal or inverted, use full shot range
                new_start = shot_start
                new_end = shot_end
                needs_fix = True

            if needs_fix:
                fix_count += 1
                logger.info(
                    f"[PostProcess] {shot.get('shot_id', '?')}: "
                    f"sub_shot_time [{orig_start}, {orig_end}] → "
                    f"[{new_start}, {new_end}] "
                    f"(shot range: [{shot_start}, {shot_end}])"
                )
                sub["sub_shot_time"] = [round(new_start, 2), round(new_end, 2)]

    if fix_count > 0:
        logger.info(
            f"[PostProcess] Fixed {fix_count} sub_shot_time entries "
            f"across {len(shots)} shots"
        )
    else:
        logger.debug("[PostProcess] All sub_shot_times within bounds, no fixes needed")

    return round_result
