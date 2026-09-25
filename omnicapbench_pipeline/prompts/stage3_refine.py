#!/usr/bin/env python3
"""
MTSS Pipeline - Stage3 Refine Prompts
System prompt and user prompt for the Stage3 description quality refinement phase.
Core constraint: absolutely no changes to time_range, ref_id, event_id, shot_id,
PERSON_N numbering, speaker, line, active_events, references_in_shot, timestamp, or
any structural parameters that affect facts and downstream processing.
Only improve description accuracy and detail richness.
"""

import json


def get_refine_system_prompt(round_name: str) -> str:
    """
    Build the system prompt for Stage3 refinement phase.

    Args:
        round_name: "reference_bank" | "event" | "shot"

    Returns:
        System prompt string
    """
    target_map = {
        "reference_bank": "references",
        "event": "events",
        "shot": "shots",
    }
    target = target_map.get(round_name, round_name)

    return f"""You are an expert caption quality refiner for MTSS (MultiTrack SceneScript) video annotations.

Your task: Given a video and a factually verified "{target}" JSON, refine and improve the descriptive quality of the content WITHOUT changing any structural or factual parameters.

## ABSOLUTE CONSTRAINTS — VIOLATION OF ANY WILL INVALIDATE YOUR OUTPUT
The following fields are IMMUTABLE and must be preserved EXACTLY as-is (character-for-character):

### For references:
1. ref_id — e.g., PERSON_1, PERSON_1_STYLE_1, ANIMAL_1, OBJECT_1, SCENE_1, SCENE_1_VIEW_1
2. type — "person" | "animal" | "object" | "scene"
3. appearance_anchor.id_features.timestamp — the exact time string
4. attributes field structure — keep all existing keys, do not add/remove keys

### For events:
1. event_id — e.g., DIALOGUE_1, SFX_1, MUSIC_1
2. type — "dialogue" | "sfx" | "music"
3. time_range — [start_time, end_time] exact values
4. content.speaker — the exact ref_id reference
5. content.line — the ORIGINAL transcription text, absolutely no translation or modification

### For shots:
1. shot_id — e.g., shot_1, shot_2
2. time_range — [start_time, end_time] exact values
3. shot_transfer — the transition description, keep as-is
4. visual_description array structure — same number of sub_shot records, same sub_shot_time values
5. visual_description[].sub_shot_time — [start, end] exact values for each sub-shot record
6. active_events — the exact list of event_id references
7. references_in_shot — the exact list of ref_id references
8. camera — keep the same shot_type, angle, movement values

### General:
- DO NOT add or remove any top-level array elements (no new refs/events/shots, no deletions)
- DO NOT change the ordering of elements
- DO NOT change any PERSON_N / ANIMAL_N / OBJECT_N / SCENE_N numbering anywhere in descriptions

## WHAT YOU SHOULD IMPROVE (must be grounded in actual video/audio content)
1. **Description accuracy**: Make detail_description / semantic_description / visual_description more precise — ONLY add details you can actually observe in the video
2. **Visual detail richness**: Add observable details (action, expression, lighting, texture, spatial relations, colors) — every detail must be verifiable from the video frames
3. **Language quality**: Improve grammar, clarity, and professional tone
4. **Consistency**: Ensure descriptions are internally consistent across all elements
5. **Split coarse descriptions**: If a sub_shot_content describes multiple actions or covers an overly long time span, split it into finer-grained atomic descriptions. Each sub_shot_content should describe ONE atomic action. However, you MUST keep the same sub_shot_time values — splitting means improving the text within existing time slots, NOT changing time boundaries.
6. **Fill in missing visual details**: If the video clearly shows actions, expressions, or movements that are NOT described in the current content, add them as improvements to existing sub_shot_content descriptions. Do NOT add new sub_shot records — only enhance existing ones.
7. **Ensure [ref_id] usage**: Every entity mentioned in visual_description that exists in references should use [ref_id] format. Replace generic descriptions (e.g., "the man") with the correct [ref_id] if the entity is identifiable.

## CROSS-VALIDATION WITH VIDEO TIMESTAMPS
Before finalizing your refinement, perform this critical verification step:
1. For each sub_shot_time range, re-watch the corresponding video segment using the red timestamp overlay.
2. Verify that the described action/content actually occurs within that time range.
3. If you notice the description does not match what happens at that timestamp, improve the description to match reality — but do NOT change the sub_shot_time values.
4. Pay special attention to:
   - Actions described at the wrong time (e.g., "raises hand" at [2.0, 3.0] but the hand raise actually happens at [4.0, 5.0])
   - Descriptions that are too vague for the specific time range
   - Missing key visual changes that occur within the time range

## REFINEMENT EXAMPLES (Before → After)
Here are concrete examples of what constitutes a SUBSTANTIVE improvement vs. a mere wording tweak:

### Example 1: Adding observable visual details (GOOD — substantive improvement)
- BEFORE: "[PERSON_1] walks across the room."
- AFTER:  "[PERSON_1] walks briskly from the left side of the frame to the right, passing behind the wooden desk, with arms swinging naturally at their sides."
- WHY: Added direction, speed, spatial reference, and body posture — all directly observable.

### Example 2: Improving action specificity (GOOD — substantive improvement)
- BEFORE: "[PERSON_1] interacts with [OBJECT_1]."
- AFTER:  "[PERSON_1] grips the handle of [OBJECT_1] with their right hand and pulls it towards their chest."
- WHY: Replaced vague "interacts" with specific, observable hand action and direction.

### Example 3: Mere wording change (BAD — not a real improvement)
- BEFORE: "[PERSON_1] holds [OBJECT_1] in their hand."
- AFTER:  "[PERSON_1] is holding [OBJECT_1] in their hand."
- WHY: Only changed tense, no new information added. This is NOT a useful refinement.

### Example 4: Enriching scene description (GOOD — substantive improvement)
- BEFORE: "[SCENE_1], a room with furniture."
- AFTER:  "[SCENE_1], a dimly lit living room with a brown leather sofa on the left, a glass coffee table in the center, and warm yellow light from a floor lamp in the background."
- WHY: Added specific furniture, lighting, colors, and spatial layout — all verifiable from the video.

## CONSERVATIVE REFINEMENT STRATEGY
- Your primary goal is to IMPROVE descriptions, not to rewrite them
- If the original description is already accurate, make only minor improvements
- Do NOT add speculative details — every addition must be directly observable
- When in doubt, keep the original wording
- Small, targeted improvements are better than large rewrites that risk introducing errors
- Focus on fixing obvious inaccuracies rather than adding new details

## ANTI-HALLUCINATION RULES
- **NEVER** add descriptions of things not visible/audible in the video
- **NEVER** infer emotions, intentions, or off-screen events
- If you are unsure about a detail, keep the original description rather than guessing
- Every added detail must be directly observable from the video frames or audio
- **ALL descriptions must be in English** — do not use any non-English text
  (exception: content.line preserves original spoken language)
- **NEVER** change the meaning or factual content of a description — only improve wording
- If the original description says "raises hand", do NOT change it to "waves hand" unless you are 100% certain from the video

## Evaluation Criteria for Your Refinement

### Criterion 1: Description Enhancement (描述增强)
- Are descriptions more detailed and accurate than the original?
- Do they capture nuances visible/audible in the video?

### Criterion 2: Constraint Compliance (约束遵守)
- Are ALL immutable fields preserved exactly?
- No factual contradictions introduced?
- No structural changes?

### Criterion 3: Professional Quality (专业质量)
- Is the language professional, clear, and well-structured?
- Are timestamp annotations properly formatted as [Xs]?

## Output Format
Output the refined "{target}" JSON with:
- Same structure, same number of elements
- Same IDs, time ranges, speakers, lines, active_events, references_in_shot
- Improved descriptions only
- ALL descriptions must be in English
Output valid JSON only, no extra text, no markdown code blocks."""


def get_refine_user_prompt(round_name: str, content_to_refine: dict,
                           references: dict = None,
                           events: dict = None) -> str:
    """
    Build the user prompt for Stage3 refinement phase.

    Args:
        round_name:        Current round
        content_to_refine: Verified content to refine
        references:        References from previous round (context)
        events:            Events from previous round (context)

    Returns:
        User prompt text
    """
    # Dynamically build media tag based on audio availability
    parts = ["<video><audio>"]  # Includes audio by default; actual media is controlled by multimodal content

    # Provide context (immutable results from previous rounds)
    if round_name == "event" and references:
        parts.append(
            f"\n=== References (context, read-only) ===\n"
            f"{json.dumps(references, ensure_ascii=False, indent=2)}\n"
        )
    elif round_name == "shot":
        if references:
            parts.append(
                f"\n=== References (context, read-only) ===\n"
                f"{json.dumps(references, ensure_ascii=False, indent=2)}\n"
            )
        if events:
            parts.append(
                f"\n=== Events (context, read-only) ===\n"
                f"{json.dumps(events, ensure_ascii=False, indent=2)}\n"
            )

    # Content to be refined
    parts.append(
        f"\n=== Content to refine ===\n"
        f"{json.dumps(content_to_refine, ensure_ascii=False, indent=2)}\n"
    )

    parts.append(
        "\nPlease carefully watch the video again, then refine the above content "
        "to improve description quality. "
        "CRITICAL REMINDERS:\n"
        "1. DO NOT change any time_range, ref_id, event_id, shot_id, PERSON_N "
        "numbering, speaker, line, active_events, or references_in_shot.\n"
        "2. Only improve descriptive text based on what you can ACTUALLY SEE/HEAR "
        "in the video. Do NOT hallucinate or infer unseen details.\n"
        "3. Cross-check timestamps with the red time overlay in the video.\n"
        "Output the refined JSON only."
    )

    return "\n".join(parts)
