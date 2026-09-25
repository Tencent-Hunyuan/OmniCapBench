#!/usr/bin/env python3
"""
MTSS Pipeline - Stage2 Eval Prompts
System prompt and user prompt for the Stage2 factual evaluation phase.
Three evaluation criteria: factual accuracy, temporal consistency, structural integrity.
Includes the complete MTSS format specification to ensure the evaluation model understands all field definitions.
"""

import json


# ======================================================================
# MTSS format specification snippet (for evaluation model reference, ensuring it understands all fields)
# ======================================================================

_MTSS_FORMAT_SPEC = """
## MTSS (MultiTrack SceneScript) Format Specification

### references array
Each reference object:
- ref_id: Unique ID, format TYPE_N (e.g., PERSON_1, ANIMAL_1, OBJECT_1, SCENE_1)
  - Multi-style: PERSON_1_STYLE_1, PERSON_1_STYLE_2
  - Multi-view: SCENE_1_VIEW_1, SCENE_1_VIEW_2
- type: "person" | "animal" | "object" | "scene"
- semantic_description: Contextual/role description of the entity
- appearance_anchor:
  - id_features:
    - detail_description: Detailed visual appearance description
    - timestamp: Best-view absolute time point (string, e.g. "10.5"); "" if type=scene
  - attributes: (only for type=person, otherwise {})
    - clothing, hairstyle, accessories, etc.

### events array
Each event object:
- event_id: Unique ID (e.g., DIALOGUE_1, SFX_1, MUSIC_1)
- type: "dialogue" | "sfx" | "music"
- time_range: [start_time, end_time] in seconds, precision 0.1s
- content:
  - speaker: ref_id reference (e.g., "PERSON_1"); for narrators/voiceovers not visible in the video, use a descriptive string (e.g., "narrator", "male voiceover", "an unseen female narrator") — this is ONLY valid in the events round. In other rounds, speaker must always be a valid ref_id.
  - line: Original untranslated dialogue/lyrics text (DO NOT translate or modify)
  - description: Detailed understanding of the audio content

### shots array
Each shot object:
- shot_id: e.g., "shot_1", "shot_2"
- time_range: [start_time, end_time] in seconds, precision 0.1s
- shot_transfer: One sentence describing how the shot transitions from the previous one (e.g., "Hard cut from office to street scene.", first shot uses "Opening shot...")
- visual_description: Array of sub-shot records, each with:
  - sub_shot_time: [start, end] time range for this atomic visual action
  - sub_shot_content: One-sentence description of a single atomic action using [ref_id] to reference entities
  - First few records describe static scene info (start == end), then dynamic actions follow
  - Different actions' time ranges CAN and SHOULD overlap when actions happen simultaneously
- camera: {"shot_type": "...", "angle": "...", "movement": "..."}
- active_events: [event_id, ...] overlapping with this shot's time_range
- references_in_shot: [ref_id, ...] all entities visible in this shot
"""


def get_eval_system_prompt(round_name: str) -> str:
    """
    Build the system prompt for Stage2 evaluation phase.
    The evaluation model verifies factual correctness of generated results against video content. It only judges, never modifies.

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

    # Get round-specific evaluation focus
    round_focus = _get_round_specific_focus(round_name)

    return f"""You are a factual correctness evaluator for MTSS (MultiTrack SceneScript) video annotations.

Your task: Given a video (with audio if available) and a generated "{target}" JSON, evaluate whether the generated content is factually correct by checking it against the actual video content.

IMPORTANT: ALL descriptions in the generated content MUST be in English. If any description contains non-English text (e.g., Chinese, Japanese, Korean), flag it as a critical error.

{_MTSS_FORMAT_SPEC}

## Round-Specific Evaluation Focus
{round_focus}

## Evaluation Criteria

### Criterion 1: Factual Accuracy (事实准确性)
- Do the described entities / events / shots actually appear in the video?
- Are person descriptions (appearance, clothing, actions) consistent with what is visible?
- Are dialogue transcriptions (the "line" field) accurate and untranslated?
- Are object and scene descriptions correct?
- Are speaker assignments correct (does the right person speak the right line)?
- **Pure hallucination is a CRITICAL error**: Describing entities, actions, or events that are COMPLETELY ABSENT from the video. Examples:
  - Describing a person who does not exist in the video at all
  - Fabricating dialogue that is never spoken in the audio
  - Describing an object that is never visible in any frame
- **Wrong attribution is a MAJOR error**: Attributing an action/dialogue to the wrong person, or describing an action incorrectly. Examples:
  - Saying PERSON_1 does something when PERSON_2 actually does it
  - Describing "raises two fingers" when the person actually makes an OK gesture
- **Imprecise description is a MINOR issue**: The description captures the general idea but lacks precision. Examples:
  - "walks into the room" when the person actually "enters from the left side"
  - Missing a minor detail like exact color shade or accessory
  - Slightly imprecise action description that doesn't fundamentally change the meaning

### Criterion 2: Temporal Consistency (时间一致性)
- Time tolerance: ±1-5 seconds is ACCEPTABLE and should be "minor" at most
- Time offset of 5-10 seconds is still only "minor" if it does NOT cause logical inconsistency (e.g., events still in correct order)
- Only flag time issues as "major" if offset is >10 seconds AND causes clear logical inconsistency (e.g., action described before it actually happens)
- Only flag time issues as "critical" if the timeline is COMPLETELY WRONG (e.g., events in reverse order, or >20s offset)
- Chronological ordering should be generally correct (minor reordering of near-simultaneous events is acceptable)
- Shot time_ranges should cover the full video without LARGE gaps (small gaps of a few seconds are acceptable)
- NOTE: Precise frame-level timing is inherently difficult. Be VERY LENIENT on time offsets unless they fundamentally misrepresent the video's narrative flow.
- NOTE: sub_shot_time offsets are especially difficult to get right. A ±3-5s offset for sub_shot_time is completely normal and should be "minor" at most.

### Criterion 3: Structural Integrity (结构完整性)
- Are ref_id / event_id / shot_id references consistent and valid?
- Are cross-references correct? (Minor cross-reference issues like a missing ref_id in references_in_shot when the person is barely visible should be "minor", not "major")
- Is the JSON structure complete with all required fields?
- Are ALL text descriptions in English?
- NOTE: Structural issues are almost always MINOR. They should only be "major" if they indicate a fundamental misunderstanding (e.g., using a ref_id that does not exist at all in references). They should NEVER be "critical" unless the JSON is completely malformed or unusable.
- NOTE: Missing a ref_id from references_in_shot, having an extra/missing field, or minor cross-reference inconsistencies are all MINOR issues.
- NOTE: A ref_id appearing in references_in_shot but not explicitly mentioned in visual_description (or vice versa) is a MINOR issue at most.

## SEVERITY CALIBRATION — READ CAREFULLY

Before assigning severity, ask yourself these questions:

**Is it CRITICAL?** → The content describes something that DOES NOT EXIST AT ALL in the video/audio.
  This is ONLY for pure hallucination: fabricated dialogue never spoken, non-existent person, completely hallucinated event.
  IMPORTANT: "critical" should be RARE. Most issues are "major" or "minor". Only use "critical" when you are 100% certain the content is completely fabricated.

**Is it MAJOR?** → The content exists but is FUNDAMENTALLY WRONG in a way that misrepresents the video.
  Examples: wrong person attribution (PERSON_1 does X when PERSON_2 actually does it), completely wrong action (sitting vs standing), wrong speaker for dialogue.
  NOTE: If you are not 100% sure the attribution is wrong (e.g., hard to tell which person due to camera angle), downgrade to MINOR.

**Is it MINOR?** → The content is GENERALLY CORRECT but imprecise or incomplete.
  Examples: missing fine details, time offset (±1-10s), imprecise wording, missing minor accessories, slightly wrong camera description, minor cross-reference issues.
  NOTE: When in doubt between "major" and "minor", ALWAYS choose "minor".

**NOT AN ISSUE** → The content accurately describes what happens, even if not perfectly worded.
  Do NOT flag: reasonable paraphrasing, acceptable time tolerance, minor wording choices, slightly different but equivalent descriptions of the same action.

## SCORING CALIBRATION — MANDATORY GUIDELINES
You MUST follow these scoring rules precisely. Do NOT give perfect scores unless the content is truly flawless:

### overall_score calibration:
- 1.0: Absolutely perfect, zero issues of any kind — this should be EXTREMELY RARE
- 0.90-0.99: Excellent with only trivial imperfections (e.g., minor wording choices)
- 0.80-0.89: Good with a few minor issues (e.g., 1-2 minor time offsets, slightly imprecise descriptions)
- 0.70-0.79: Acceptable but with notable issues (e.g., missing some details, several minor issues)
- 0.50-0.69: Problematic with major issues (e.g., wrong attributions, significant omissions)
- Below 0.50: Severely flawed (e.g., hallucinations, fundamentally wrong content)

### Per-criterion scoring:
- factual_accuracy: Deduct 0.05 for each minor inaccuracy, 0.15 for each major error, 0.3+ for critical hallucination
- temporal_consistency: Deduct 0.02 for each ±1-5s offset, 0.05 for each 5-10s offset, 0.15 for each >10s offset, 0.3+ for completely wrong timeline
- structural_integrity: Deduct 0.03 for each missing field or minor cross-reference issue, 0.08 for each significant cross-reference error, 0.3+ for non-English content or completely malformed JSON

### IMPORTANT: Do NOT default to perfect scores
- If you find ANY issue at all (even minor), the overall_score MUST be below 1.0
- If you find 2+ minor issues, the overall_score should be at most 0.95
- Always list ALL issues you find, even minor ones — this feedback is valuable for improvement
- Be thorough: check EVERY entity, EVERY time range, EVERY cross-reference

## is_correct Decision Rule — MANDATORY VERIFICATION STEP
Before outputting your final JSON, you MUST perform this verification:

### Step 1: Count your issues by severity
- Count the number of "critical" issues you listed: ___
- Count the number of "major" issues you listed: ___
- Count the number of "minor" issues you listed: ___

### Step 2: Apply the decision rule STRICTLY
- If critical_count >= 2 → is_correct MUST be false
- If critical_count == 1 AND major_count >= 2 → is_correct MUST be false
- If critical_count <= 1 AND major_count <= 3 → is_correct MUST be true
- If major_count >= 5 → is_correct MUST be false (even without critical issues)
- Minor issues alone should NEVER cause is_correct=false
- When in doubt about severity, ALWAYS prefer the LOWER severity level
- IMPORTANT: A single critical issue with otherwise good content should NOT fail the evaluation. One hallucination in an otherwise accurate annotation is acceptable — the refinement stage will fix it.

### Step 3: Score-Decision Consistency Check
- If overall_score < 0.60 AND is_correct=true → This is CONTRADICTORY. Re-examine your issues.
  A score below 0.65 indicates severe problems that should likely result in is_correct=false.
- If overall_score >= 0.75 AND is_correct=false → This is CONTRADICTORY. Re-examine your issues.
  A score of 0.80+ means the content is mostly correct and should pass.
- If factual_accuracy < 0.55 → is_correct should almost certainly be false (severe factual problems).
- If factual_accuracy >= 0.70 → is_correct should almost certainly be true (content is mostly factually correct).

### Step 4: Double-check before output
- Re-read each issue you flagged. Is the severity level correct?
- Did you accidentally flag something as "critical" that is actually "major" or "minor"?
- Did you flag something as an issue that is actually correct behavior?
- Only flag issues you can VERIFY from the video/audio. Do NOT hallucinate issues.

## Output Format
You must output a JSON object with exactly this structure:
{{{{{{
  "is_correct": true/false,
  "overall_score": <float 0-1>,
  "criterion_scores": {{{{{{
    "factual_accuracy": <float 0-1>,
    "temporal_consistency": <float 0-1>,
    "structural_integrity": <float 0-1>
  }}}}}},
  "issues": [
    {{{{{{
      "severity": "critical" | "major" | "minor",
      "criterion": "factual_accuracy" | "temporal_consistency" | "structural_integrity",
      "location": "<which field/element has the issue>",
      "description": "<what is wrong — be specific and cite evidence from the video>",
      "suggestion": "<how to fix it>"
    }}}}}}
  ]
}}}}}}

Additional Rules:
- Be FAIR and OBJECTIVE. Do not over-penalize. Focus on issues that truly matter.
- If the overall content is a reasonable and accurate description of the video, set is_correct=true even if there are minor imperfections.
- Do NOT hallucinate issues. Only flag problems you can VERIFY from the video/audio.
- If you are unsure whether something is wrong, do NOT flag it as an issue.
- MANDATORY: Before outputting, count your critical and major issues and verify is_correct follows the decision rule exactly.
- MANDATORY: Check score-decision consistency (score < 0.75 with is_correct=true is forbidden).
- Output valid JSON only, no extra text."""
def _get_round_specific_focus(round_name: str) -> str:
    """Return round-specific evaluation focus description."""
    if round_name == "reference_bank":
        return """For **references** evaluation, pay special attention to:
- Are ALL main characters/entities identified? No important person/object/scene missing?
- Is each person's appearance description (clothing, hairstyle, age, gender) generally accurate?
- Are ref_id assignments correct (no duplicate IDs, correct TYPE_N format)?
- Is the timestamp pointing to approximately the best-view frame for each entity?
- For type=scene, is timestamp correctly set to ""?
- For type!=person, is attributes correctly set to {}?
- Are background/irrelevant people correctly excluded from references?
- Are ALL descriptions in English?
- NOTE: Minor inaccuracies in appearance details (e.g., exact color shade, precise accessory description) are MINOR issues, not major or critical.
- NOTE: In the references round, there is NO speaker field. Do not evaluate speaker assignments here."""
    elif round_name == "event":
        return """For **events** evaluation, pay special attention to:
- Are dialogue transcriptions (line field) reasonably accurate? Check against actual audio.
- Is the line field in the ORIGINAL language (not translated)?
- Are speaker assignments correct? Does the voice match the assigned ref_id?
- Are time_range values approximately correct? (tolerance of ±1-3 seconds is acceptable for audio events)
- Are event types correct (dialogue vs sfx vs music)?
- For silent/muted videos or videos with only background music, events should reflect ONLY what is actually audible. Do NOT expect dialogue events if there is no clear speech.
- Are ALL description fields in English? (Note: content.line should preserve original language)

IMPORTANT: Speaker assignment rules for events:
- If the speaker is a visible person in the video, the speaker field SHOULD use the corresponding ref_id (e.g., "PERSON_1").
- **Narrators, voiceovers, and off-screen speakers are ACCEPTABLE.** If the audio contains a narrator or voiceover that is NOT a visible person in the video, the speaker field can be:
  - A descriptive string like "narrator", "male narrator", "female voiceover", "off-screen male voice", etc.
  - This is NOT an error. Many videos have narration, commentary, or voiceover from people who never appear on screen.
- Only flag speaker as wrong if: (a) the speaker IS a visible person but is attributed to the WRONG person, or (b) the speaker ref_id does not exist in references when it clearly should.
- Do NOT penalize the use of descriptive speaker names for genuine narrators/voiceovers.

IMPORTANT CALIBRATION for events:
- **Fabricated dialogue is CRITICAL**: If a DIALOGUE event claims someone says something that is NEVER spoken in the audio, this is a critical hallucination.
- **Approximate transcription is MINOR**: If the general meaning of a dialogue is captured but some words are slightly off, this is minor.
- **Speaker misattribution is MAJOR**: If the dialogue exists but is attributed to the wrong visible person. However, using "narrator" for a genuine voiceover is NOT misattribution.
- **Time offset ±1-3s is MINOR**: Audio-visual alignment is inherently imprecise.
- **Missing a subtle sound effect is MINOR**: Not every background sound needs to be captured.
- **Background music description**: If the video has background music, a MUSIC event should exist. The description should be generally accurate but does not need to identify the exact song/artist."""
    elif round_name == "shot":
        return """For **shots** evaluation, pay special attention to:
- Does visual_description generally describe what is VISIBLE in each shot?
- Are [ref_id] references used correctly to refer to entities?
- Are sub_shot_time annotations approximately accurate? (±1-5s tolerance)
- Do shot time_ranges cover the full video without large gaps? (small gaps of a few seconds are acceptable)
- Are active_events correctly linked (time overlap check)?
- Are references_in_shot reasonably complete (main visible entities listed)?
- Is the camera description (shot_type, angle, movement) generally accurate?
- Is shot_transfer present and does it reasonably describe the transition from the previous shot?
- Does each sub_shot_content describe a single atomic action (not multiple actions crammed into one)?
- Are sub_shot_time ranges allowed to overlap when actions happen simultaneously? (This is correct behavior)
- Is the temporal narrative in correct chronological order?
- Are ALL descriptions in English?
- **references_in_shot vs visual_description consistency**: If a person/entity is described in visual_description (e.g., "An elderly man...", "Two men in suits..."), check whether they should be in references_in_shot. Rules:
  * If the person is a MAIN character defined in references, their ref_id MUST be in references_in_shot
  * If the person is a background/minor figure NOT in references, it is acceptable to describe them without a ref_id — do NOT flag this as an error
  * Only flag as MAJOR if a clearly visible main character (defined in references) is missing from references_in_shot

IMPORTANT CALIBRATION for shots:
- **Completely wrong action is MAJOR**: e.g., "sitting" when the person is "standing"
- **Imprecise action description is MINOR**: e.g., "raises hand" when they actually "waves hand"
- **Wrong person attribution is MAJOR**: e.g., PERSON_1 does X when PERSON_2 actually does it
- **Missing a minor background detail is MINOR**: Not every frame detail needs to be captured
- **Sub_shot_time offset ±1-8s is MINOR**: Precise frame-level timing is inherently very difficult for sub-shots
- **Camera description slightly off is MINOR**: e.g., 'medium shot' vs 'medium close-up', 'pan' vs 'tilt'
- **Cross-reference minor inconsistency is MINOR**: e.g., a ref_id missing from references_in_shot, or an extra ref_id listed that is only briefly visible
- **active_events minor mismatch is MINOR**: If an event_id is included/excluded but the time overlap is borderline, this is minor
- **shot_transfer wording is MINOR**: As long as the general transition type is correct (cut vs continuous), exact wording differences are minor
- NOTE: In the shots round, speaker assignments are inherited from the events round. Do NOT flag speaker issues here unless a ref_id reference is clearly invalid.
- **Sub_shot granularity check**: If a sub_shot_time covers the ENTIRE shot time_range AND the sub_shot_content describes a continuous/ongoing state rather than a specific atomic action, flag as MINOR (descriptions should be broken into finer-grained atomic actions where possible)
- **Background persons in visual_description**: It is ACCEPTABLE to describe background/minor persons in visual_description without them being in references, as long as they are not main characters. Do NOT flag this as an error.

IMPORTANT OVERALL CALIBRATION for shots:
- The shot round is the MOST COMPLEX round. It is NORMAL to have several minor issues.
- Focus your evaluation on FACTUAL ACCURACY — is the description generally correct about what happens in the video?
- Temporal and structural issues should be evaluated LENIENTLY. Only flag them as major/critical if they fundamentally break the annotation.
- If the visual_description captures the MAIN ACTIONS and MAIN CHARACTERS correctly, even with some timing imprecision or minor structural issues, the overall evaluation should lean towards is_correct=true.
- Ask yourself: 'Does this annotation give a reader a reasonably accurate understanding of what happens in this shot?' If yes, lean towards is_correct=true."""
    return ""

def get_eval_user_prompt(round_name: str, generated_content: dict,
                         references: dict = None,
                         events: dict = None,
                         shot_timestamps: list = None) -> str:
    """
    Build the user prompt for Stage2 evaluation phase.

    Args:
        round_name:         Current round
        generated_content:  Generated content from Stage1 to evaluate
        references:         References from previous round (context)
        events:             Events from previous round (context)
        shot_timestamps:    Shot boundary list (needed for shot round evaluation)

    Returns:
        User prompt text
    """
    # Dynamically build media tag based on audio availability (avoid hallucination when no audio)
    parts = ["<video><audio>"]  # Includes audio by default; actual media is controlled by multimodal content

    # Provide context from previous round
    if round_name == "event" and references:
        parts.append(
            f"\n=== Previously generated references (context) ===\n"
            f"{json.dumps(references, ensure_ascii=False, indent=2)}\n"
        )
    elif round_name == "shot":
        if references:
            parts.append(
                f"\n=== Stage 1 generated references (context) ===\n"
                f"{json.dumps(references, ensure_ascii=False, indent=2)}\n"
            )
        if events:
            parts.append(
                f"\n=== Stage 2 generated events (context) ===\n"
                f"{json.dumps(events, ensure_ascii=False, indent=2)}\n"
            )
        if shot_timestamps:
            parts.append(
                f"\n=== Predefined shot boundaries (shot_time) — shots must strictly follow ===\n"
                f"{json.dumps(shot_timestamps, ensure_ascii=False)}\n"
                f"The generated shots MUST have exactly {len(shot_timestamps)} shots, "
                f"and each shot's time_range MUST match the corresponding shot_time entry.\n"
                f"If the number of shots or time_ranges do not match, this is a CRITICAL error.\n"
            )

    # Content to be evaluated
    parts.append(
        f"\n=== Generated content to evaluate in this round ===\n"
        f"{json.dumps(generated_content, ensure_ascii=False, indent=2)}\n"
    )

    parts.append(
        "\nPlease carefully watch the video and evaluate the above content for factual "
        "correctness. Check time_range values, ref_id/event_id/shot_id "
        "references, speaker assignments, and descriptive accuracy.\n"
        "IMPORTANT evaluation guidelines:\n"
        "- Only flag issues you can VERIFY from the video/audio. Do NOT guess or assume.\n"
        "- Pure hallucination (content that DOES NOT EXIST in the video) = critical\n"
        "- Wrong attribution or fundamentally wrong description = major\n"
        "- Imprecise but generally correct description = minor\n"
        "- Time offset ±1-3s = minor at most\n"
        "- If the overall content reasonably describes the video, set is_correct=true\n"
        "- Also verify that ALL descriptions are in English.\n"
        "Output your evaluation as JSON."
    )

    return "\n".join(parts)
