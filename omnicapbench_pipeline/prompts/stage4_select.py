#!/usr/bin/env python3
"""MTSS Pipeline - Stage4 Select Prompts
System prompt and user prompt for the Stage4 atomic-unit selection phase.
Selects the optimal combination from up to 3 candidate captions at atomic-unit granularity,
without making any modifications.

Candidate sources:
  - Candidate A: original verified caption before refinement (Stage2-passed version)
  - Candidate B: refined caption (Stage3 model 1 refined version)
  - Candidate C: refined caption (Stage3 model 2 refined version)
  Note: If B or C failed Stage2 re-eval, that candidate will not be provided.

Atomic-unit granularity:
  - references: each reference object is one atomic unit
  - events: each event object is one atomic unit
  - shots: sub-components within each shot can be selected independently:
    * visual_description as a whole
    * camera as a whole
    * But time_range, shot_id, active_events, references_in_shot must remain consistent
    * Each action/description segment within a shot can also be independently selected
"""
import json


def get_select_system_prompt(round_name: str) -> str:
    """
    Build the system prompt for Stage4 selection phase.

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

    return f"""You are an expert selector for MTSS (MultiTrack SceneScript) video annotations.

Your task: Given a video and UP TO THREE candidate versions of a "{target}" JSON, select the BEST atomic units from each candidate and assemble the final result.

The candidates are:
- **Candidate A**: The original verified caption (passed Stage2 evaluation, before any refinement)
- **Candidate B**: Refined version from refinement model 1 (if available)
- **Candidate C**: Refined version from refinement model 2 (if available)

Note: Sometimes only 2 candidates are provided (A and B) if one refinement failed evaluation.
Candidate A is always the "safe" baseline — it passed factual verification but may lack descriptive richness.
Candidates B and C are refined versions — they may have richer descriptions but could also introduce subtle inaccuracies.

## CRITICAL RULES

### Rule 1: Atomic Unit Selection
For each element, compare the corresponding elements from ALL available candidates, and pick the best one at the finest granularity:

**For references:**
- Each reference object (identified by ref_id) is an atomic unit
- Compare: semantic_description, detail_description, attributes
- Select the version with more accurate and detailed descriptions

**For events:**
- Each event object (identified by event_id) is an atomic unit
- Compare: content.description (the understanding/analysis part)
- Note: content.line MUST be identical across all candidates (original transcription)
- Select the version with better audio analysis

**For shots:**
- Within each shot (identified by shot_id), sub-components can be selected independently:
  * shot_transfer: pick the best transition description from any candidate
  * visual_description: an array of sub_shot records; each sub_shot_content can be independently selected from any candidate (sub_shot_time must be identical)
  * camera: pick the best one
  * IMMUTABLE (must be same across all): time_range, shot_id, active_events, references_in_shot, sub_shot_time values

### Rule 2: NO MODIFICATION — VERBATIM SELECTION ONLY
You MUST select text verbatim from one of the candidates. You are absolutely forbidden from:
- Rewriting or paraphrasing any text
- Merging parts of different candidates within a single atomic unit
- Creating new descriptions
- Changing any time_range, ref_id, event_id, shot_id, PERSON_N numbering, speaker, line, active_events, or references_in_shot

### Rule 3: Selection Criteria

#### Criterion 1: Accuracy (准确性) — HIGHEST PRIORITY
- Which candidate's description better matches the actual video content?
- Are visual details (colors, positions, actions) more accurate?
- When in doubt between accuracy and richness, ALWAYS prefer accuracy.
- Candidate A is the factually verified baseline — prefer it over B/C if B/C introduced any inaccuracy.

#### Criterion 2: Detail Richness (细节丰富度)
- Which candidate provides more useful visual/audio details?
- Are descriptions more specific and informative?

#### Criterion 3: Clarity (清晰度)
- Which candidate's language is clearer and more professional?
- Is the narrative flow better structured?

## Output Format
Output a JSON object with:
{{
  "selections": [
    {{
      "element_id": "<ref_id/event_id/shot_id>",
      "chosen": "A" | "B" | "C",
      "sub_selections": {{
        "visual_description": "A" | "B" | "C",
        "camera": "A" | "B" | "C"
      }},
      "reason": "<brief reason for selection>"
    }}
  ],
  "final_{target}": [ ... ]
}}

Notes:
- "sub_selections" is only needed for shots; omit for references and events
- "final_{target}" must contain the assembled result using your selected atomic units
- The final result must be valid JSON with all required MTSS fields
- ALL descriptions must be in English
- Output valid JSON only, no extra text, no markdown code blocks."""


def get_select_user_prompt(round_name: str,
                           candidate_a: dict,
                           candidate_b: dict,
                           candidate_c: dict = None,
                           references: dict = None,
                           events: dict = None) -> str:
    """
    Build the user prompt for Stage4 selection phase.

    Args:
        round_name:   Current round
        candidate_a:  Original verified caption (before refinement)
        candidate_b:  Refined caption 1 (Stage3 model 1)
        candidate_c:  Refined caption 2 (Stage3 model 2), optional
        references:   References from previous round (context)
        events:       Events from previous round (context)

    Returns:
        User prompt text
    """
    # Dynamically build media tag based on audio availability
    parts = ["<video><audio>"]  # Includes audio by default; actual media is controlled by multimodal content

    # Provide context
    if round_name == "event" and references:
        parts.append(
            f"\n=== References (context reference) ===\n"
            f"{json.dumps(references, ensure_ascii=False, indent=2)}\n"
        )
    elif round_name == "shot":
        if references:
            parts.append(
                f"\n=== References (context reference) ===\n"
                f"{json.dumps(references, ensure_ascii=False, indent=2)}\n"
            )
        if events:
            parts.append(
                f"\n=== Events (context reference) ===\n"
                f"{json.dumps(events, ensure_ascii=False, indent=2)}\n"
            )

    # Candidate A: original verified (pre-refine)
    parts.append(
        f"\n=== Candidate A (Original verified, before refinement) ===\n"
        f"{json.dumps(candidate_a, ensure_ascii=False, indent=2)}\n"
    )
    # Candidate B: refined version 1
    parts.append(
        f"\n=== Candidate B (Refined version 1) ===\n"
        f"{json.dumps(candidate_b, ensure_ascii=False, indent=2)}\n"
    )
    # Candidate C: refined version 2 (optional)
    if candidate_c is not None:
        parts.append(
            f"\n=== Candidate C (Refined version 2) ===\n"
            f"{json.dumps(candidate_c, ensure_ascii=False, indent=2)}\n"
        )
        num_candidates = 3
    else:
        num_candidates = 2

    parts.append(
        f"\nYou have {num_candidates} candidates to compare. "
        "Please carefully watch the video, then compare ALL candidates "
        "against the actual video content. Select the best atomic units from "
        "any candidate, and assemble the final result. "
        "Remember: select VERBATIM from the candidates, do NOT rewrite or merge. "
        "Do NOT change any time_range, IDs, or structural parameters. "
        "Candidate A is the factually verified baseline — prefer it if B/C introduced inaccuracies. "
        "When in doubt, prefer the description that is more factually accurate "
        "over the one that is more detailed but potentially hallucinated. "
        "Output your selection and final result as JSON."
    )

    return "\n".join(parts)
