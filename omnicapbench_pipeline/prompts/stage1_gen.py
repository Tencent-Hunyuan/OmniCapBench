#!/usr/bin/env python3
"""
MTSS Pipeline - Stage1 Gen Prompts
Build system prompts and user prompts for the Stage1 generation phase.
3 rounds: reference_bank / event / shot
Strictly uses original mtss_generate_prompts/*.txt files as system prompts.
New prompt split into 3 independent files: ref_system_prompt.txt / event_system_prompt.txt / shot_system_prompt.txt
The shot prompt contains three placeholders: {references}, {events}, {shot_time}.
"""

import json
import os
import re
import logging

from .. import config

logger = logging.getLogger("omnicapbench_pipeline")

# ---------------------------------------------------------------------------
# Load original prompt files (reload on each call to ensure latest version)
# ---------------------------------------------------------------------------
_ORIGINAL_DIR = config.ORIGINAL_PROMPT_DIR


def _load_original(filename: str) -> str:
    """Load original prompt file; returns empty string if not found."""
    path = os.path.join(_ORIGINAL_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"[Stage1-Gen] Prompt file not found: {path}")
        return ""


# ======================================================================
# Public interface: get Stage1 Gen system prompt
# ======================================================================

def get_gen_system_prompt(round_name: str,
                          references: dict = None,
                          events: dict = None,
                          shot_timestamps: list = None,
                          feedback: str = None) -> str:
    """
    Build the system prompt for Stage1 generation phase.

    Args:
        round_name:      "reference_bank" | "event" | "shot"
        references:      References output from previous round (needed for event/shot rounds)
        events:          Events output from previous round (needed for shot round)
        shot_timestamps: Shot boundary list (needed for shot round, replaces {shot_time} placeholder)
        feedback:        Feedback from Stage2 eval (injected on retry)

    Returns:
        Complete system prompt string
    """
    if round_name == "reference_bank":
        prompt = _build_ref_prompt(feedback)
    elif round_name == "event":
        prompt = _build_event_prompt(references, feedback)
    elif round_name == "shot":
        prompt = _build_shot_prompt(references, events, shot_timestamps, feedback)
    else:
        raise ValueError(f"Unknown round: {round_name}")
    return prompt


def get_gen_user_prompt(round_name: str, has_audio: bool,
                        shot_timestamps: list = None) -> str:
    """
    Build the user prompt for Stage1 generation phase (appended after video/audio content).
    Note: New prompt files already contain complete instructions at the end; only media tags are provided here.

    Args:
        round_name:      Current round name
        has_audio:        Whether audio is available
        shot_timestamps:  Shot boundary list (used in shot round, but already injected in system prompt)

    Returns:
        User prompt text string
    """
    # The new prompt files already contain complete instructions at the end, user prompt only needs media tags
    # Avoid duplicate instructions confusing the model
    if has_audio:
        return (
            "Please carefully analyze the provided video and audio, then output the JSON result.\n"
            "IMPORTANT RULES:\n"
            "1. ALL descriptions MUST be in English (content.line preserves original spoken language).\n"
            "2. ANTI-HALLUCINATION: Only describe what you can DIRECTLY SEE in the video frames "
            "or DIRECTLY HEAR in the audio. Do NOT guess, infer, or fabricate any content.\n"
            "3. For dialogue: Only transcribe words you can CLEARLY HEAR. If audio is unclear, "
            "use approximate transcription or skip the event entirely.\n"
            "4. For visual descriptions: Only describe actions/objects you can CLEARLY SEE. "
            "Cross-check timestamps with the red time overlay in the video.\n"
            "5. It is better to output FEWER but CORRECT items than many items with errors.\n"
            "6. Output valid JSON only, no extra text, no markdown code blocks.\n"
            "7. NEVER describe screens as 'going dark/black' unless you CLEARLY SEE a completely black screen.\n"
            "8. NEVER describe a 'Hard cut' between shots unless there is a clear visual discontinuity. "
            "If the camera movement is continuous, use 'Continuous shot' instead.\n"
            "9. VERIFY EACH PERSON: Before writing [PERSON_X] does something, CONFIRM that person is "
            "actually visible and performing that action at that specific time. If you cannot clearly "
            "identify which person is doing what, describe the action without a ref_id.\n"
            "10. VERIFY EACH ACTION: Before describing an action, ask yourself: 'Can I clearly see this "
            "happening in the video at this exact time?' If the answer is no, do NOT include it.\n"
            "11. PREFER SIMPLE DESCRIPTIONS: Use simple, concrete descriptions of what you see. "
            "Avoid complex interpretations or assumptions about intent/emotion unless clearly visible."
        )
    else:
        return (
            "Please carefully analyze the provided video (no audio available), then output the JSON result.\n"
            "IMPORTANT RULES:\n"
            "1. ALL descriptions MUST be in English.\n"
            "2. Since there is NO AUDIO, the events array should contain ONLY visual sound effects "
            "(SFX) or background music (MUSIC) if clearly implied by visual cues. "
            "Do NOT fabricate any DIALOGUE events.\n"
            "3. ANTI-HALLUCINATION: Only describe what you can DIRECTLY SEE in the video frames.\n"
            "4. It is better to output FEWER but CORRECT items than many items with errors.\n"
            "5. Output valid JSON only, no extra text, no markdown code blocks.\n"
            "6. NEVER describe screens as 'going dark/black' unless you CLEARLY SEE a completely black screen.\n"
            "7. NEVER describe a 'Hard cut' between shots unless there is a clear visual discontinuity."
        )


# ======================================================================
# Internal prompt construction functions
# ======================================================================

def _extract_banned_content(feedback: str) -> str:
    """
    Extract hallucinated/critical content from feedback to build a BANNED CONTENT list.
    This explicitly tells the model what NOT to output again.
    """
    if not feedback:
        return ""
    banned_items = []
    # Extract critical and major issues from feedback
    lines = feedback.split("\n")
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith("[critical]") or line_stripped.startswith("[major]"):
            # Extract the description part after the location info
            # Pattern: [severity] at <location>: <description>
            match = re.match(r'\[(critical|major)\]\s+at\s+.+?:\s+(.+)', line_stripped)
            if match:
                desc = match.group(2).strip()
                banned_items.append(f"  - {desc}")
    if not banned_items:
        return ""
    return (
        "\n\n=== ★★★ BANNED CONTENT — DO NOT OUTPUT ANY OF THE FOLLOWING ★★★ ===\n"
        "The following specific content was identified as HALLUCINATED or FUNDAMENTALLY WRONG \n"
        "in your previous output. You MUST NOT include any of these in your new output:\n\n"
        + "\n".join(banned_items) + "\n\n"
        "If you are tempted to describe any of the above, STOP and SKIP it entirely.\n"
        "It is MUCH BETTER to omit content than to repeat a known hallucination.\n"
    )


def _inject_feedback(prompt: str, feedback: str) -> str:
    """Inject feedback before the final instruction marker in the prompt."""
    if not feedback:
        return prompt

    # Build BANNED CONTENT list from critical/major issues
    banned_block = _extract_banned_content(feedback)

    feedback_block = (
        f"\n\n=== CRITICAL: Feedback from previous evaluation ===\n"
        f"Your previous output was evaluated and the following issues were found.\n"
        f"You MUST fix ALL of these issues in your next generation:\n\n"
        f"{feedback}\n\n"
        f"{banned_block}"
        f"=== Key instructions for re-generation ===\n"
        f"1. Carefully read each issue above and understand what went wrong.\n"
        f"2. For hallucination errors: REMOVE the hallucinated content entirely. "
        f"Do NOT try to fix hallucinated content — just remove it.\n"
        f"3. For wrong attribution errors: Verify by carefully watching the video "
        f"which person/entity actually performs the action or speaks the line.\n"
        f"4. For time range errors: Cross-check with the red timestamp overlay in the video.\n"
        f"5. Only describe what you can DIRECTLY OBSERVE in the video/audio. "
        f"If you are unsure about something, do NOT include it.\n"
        f"6. It is better to output FEWER but CORRECT items than many items with errors.\n"
        f"7. Review the BANNED CONTENT list above — these are KNOWN hallucinations. "
        f"Do NOT repeat them under any circumstances.\n"
    )
    insert_marker = "现在严格遵循以上所有核心原则"
    if insert_marker in prompt:
        idx = prompt.rfind(insert_marker)
        return prompt[:idx] + feedback_block + "\n" + prompt[idx:]
    else:
        return prompt + feedback_block


def _build_ref_prompt(feedback: str = None) -> str:
    """Build the system prompt for the reference_bank round."""
    prompt = _load_original("ref_system_prompt.txt")
    if not prompt:
        prompt = _FALLBACK_REF_PROMPT
    return _inject_feedback(prompt, feedback)


def _build_event_prompt(references: dict = None,
                        feedback: str = None) -> str:
    """Build the system prompt for the event round, injecting previous round references."""
    prompt = _load_original("event_system_prompt.txt")
    if not prompt:
        prompt = _FALLBACK_EVENT_PROMPT

    # Replace {references} placeholder
    if references:
        ref_str = json.dumps(references, ensure_ascii=False, indent=2)
    else:
        ref_str = "[]"

    if "{references}" in prompt:
        prompt = prompt.replace("{references}", ref_str)
    else:
        # Placeholder missing from original file — append context block
        logger.warning(
            "[Stage1-Gen] event_system_prompt.txt missing {references} placeholder, "
            "appending context block to prompt tail"
        )
        context_block = (
            "\n\n=== 第一阶段输出：references（你必须基于以下实体进行标注） ===\n"
            f"{ref_str}\n"
        )
        insert_marker = "现在严格遵循以上所有核心原则"
        if insert_marker in prompt:
            idx = prompt.rfind(insert_marker)
            prompt = prompt[:idx] + context_block + "\n" + prompt[idx:]
        else:
            prompt += context_block

    return _inject_feedback(prompt, feedback)


def _build_shot_prompt(references: dict = None, events: dict = None,
                       shot_timestamps: list = None,
                       feedback: str = None) -> str:
    """Build the system prompt for the shot round, injecting references, events and shot_time from previous rounds."""
    prompt = _load_original("shot_system_prompt.txt")
    if not prompt:
        prompt = _FALLBACK_SHOT_PROMPT

    # Prepare JSON strings
    if references:
        ref_str = json.dumps(references, ensure_ascii=False, indent=2)
    else:
        ref_str = "[]"
    if events:
        evt_str = json.dumps(events, ensure_ascii=False, indent=2)
    else:
        evt_str = "[]"
    if shot_timestamps:
        shot_time_str = json.dumps(shot_timestamps, ensure_ascii=False)
    else:
        shot_time_str = "[]"

    # Try placeholder replacement first
    has_ref_ph = "{references}" in prompt
    has_evt_ph = "{events}" in prompt
    has_st_ph = "{shot_time}" in prompt

    if has_ref_ph:
        prompt = prompt.replace("{references}", ref_str)
    if has_evt_ph:
        prompt = prompt.replace("{events}", evt_str)
    if has_st_ph:
        prompt = prompt.replace("{shot_time}", shot_time_str)

    # If ANY placeholder was missing, append the full context block
    if not has_ref_ph or not has_evt_ph or not has_st_ph:
        missing = []
        if not has_ref_ph:
            missing.append("{references}")
        if not has_evt_ph:
            missing.append("{events}")
        if not has_st_ph:
            missing.append("{shot_time}")
        logger.warning(
            f"[Stage1-Gen] shot_system_prompt.txt missing placeholders: "
            f"{', '.join(missing)}, appending context block to prompt tail"
        )
        context_block = (
            "\n\n=== ★★★ 前两个阶段的输出结果（你必须严格基于以下内容进行标注） ★★★ ===\n"
            "\n=== 第一阶段输出：references ===\n"
            "You MUST use the following references to identify entities in your shots output.\n"
            "Do NOT invent new ref_ids or use ref_ids from examples — ONLY use the ones below:\n"
            f"{ref_str}\n"
            "\n=== 第二阶段输出：events ===\n"
            "You MUST reference the following events in active_events fields.\n"
            "Do NOT fabricate event_ids — ONLY use the ones below:\n"
            f"{evt_str}\n"
            "\n=== 预切分的镜头时间段（shot_time） ===\n"
            "You MUST generate exactly one shot for each time segment below:\n"
            f"{shot_time_str}\n"
        )
        insert_marker = "现在严格遵循以上所有核心原则"
        if insert_marker in prompt:
            idx = prompt.rfind(insert_marker)
            prompt = prompt[:idx] + context_block + "\n" + prompt[idx:]
        else:
            prompt += context_block

    # Inject additional constraints for atomic action granularity, person reference rules,
    # anti-hallucination examples, and continuous vs hard cut guidance
    _SHOT_EXTRA_CONSTRAINTS = """

=== ADDITIONAL CRITICAL CONSTRAINTS (MUST FOLLOW) ===

★★★ CONSTRAINT 1: sub_shot_time MUST NOT span the entire shot ★★★
- FORBIDDEN: Using the full shot time_range (e.g., [5.96, 20.68]) as sub_shot_time for action descriptions.
  This means you are NOT breaking down the actions into atomic units.
- EXCEPTION: Only the first 1-2 records (static scene/environment description) may use the full shot time_range.
- For ALL dynamic actions (movements, gestures, expressions, speaking), sub_shot_time MUST be a SHORT time segment
  (typically 0.5s-3s) that precisely captures WHEN that specific action occurs.
- BAD example: sub_shot_time=[0.0, 15.0], content="[PERSON_1] speaks continuously" ← TOO COARSE
- GOOD example: Break into multiple records:
  * [0.0, 2.5] "[PERSON_1] begins speaking with hands resting on the table"
  * [2.5, 4.0] "[PERSON_1] raises right hand to gesture"
  * [4.0, 6.0] "[PERSON_1] lowers hand back to table"

★★★ CONSTRAINT 2: Persons in visual_description vs references_in_shot ★★★
- If a person/entity described in visual_description is a MAIN character defined in references,
  you MUST use their [ref_id] (e.g., [PERSON_1]) and include them in references_in_shot.
- If a person is a background/minor figure NOT defined in references, describe them with
  generic text (e.g., "An elderly man in a brown coat") WITHOUT using [ref_id] brackets.
  They do NOT need to be in references_in_shot.
- NEVER use [ref_id] for a person who is NOT in references.

★★★ CONSTRAINT 3: ANTI-HALLUCINATION — MANDATORY VERIFICATION PROTOCOL ★★★

Before writing EACH sub_shot_content entry, you MUST perform this 3-step verification:

**Step A: PERSON VERIFICATION** — For each [PERSON_X] you mention:
  - Can you clearly see this specific person in the video at this exact time?
  - Can you distinguish them from other people? (If not, do NOT use a ref_id)
  - Is this person actually performing the described action? (Watch carefully!)
  - COMMON MISTAKE: Confusing PERSON_1 with PERSON_2 when they look similar or when
    the camera angle makes it hard to tell. If unsure, describe the action without ref_id.

**Step B: ACTION VERIFICATION** — For each action you describe:
  - Can you clearly see this action happening in the video frames?
  - Is the action happening at the time you specified? (Check the red timestamp overlay)
  - Are you describing what you SEE, not what you EXPECT to see?
  - COMMON MISTAKE: Describing "bends down" when the person actually "turns around",
    or "walks away" when they actually "stands still". Watch the actual motion carefully.

**Step C: EXISTENCE VERIFICATION** — For each entity/object you mention:
  - Does this entity actually appear in the video at this time?
  - If a person exits the frame, do NOT describe them as still present.
  - If a new person enters, verify they are actually a new person vs an existing reference.
  - COMMON MISTAKE: Describing a person who has already left the scene as still being there,
    or misidentifying a new person as an existing reference.

HIGH-FREQUENCY hallucination patterns to AVOID:

1. **"Screen goes dark/black" hallucination**: Do NOT describe a phone/monitor/TV screen
   as "going dark", "turning black", or "turning off" unless you can CLEARLY SEE the screen
   becoming completely black in the video frames.

2. **"Wrong person attribution"**: This is the #1 most common error. Before writing
   "[PERSON_X] does Y", VERIFY that it is indeed PERSON_X and not someone else.
   If you cannot clearly identify the person, write "A person does Y" instead.

3. **"Fabricated dialogue" hallucination**: Do NOT create dialogue events for sounds that
   are actually music, ambient noise, or unclear audio.

4. **"Non-existent action" hallucination**: Do NOT describe actions you cannot clearly see.
   If the video is blurry, fast-moving, or the person is partially occluded, describe only
   what you can confirm. It is MUCH BETTER to write fewer, correct descriptions.

5. **"Person still present" hallucination**: When a person exits the frame, they are GONE.
   Do NOT describe them as still being in the scene unless they clearly re-enter.

6. **Time-shifted descriptions**: Cross-check with the red timestamp overlay. If action X
   happens at 3.5s, do NOT place it at 1.0s.

GOLDEN RULE: When in doubt, OMIT. A missing description is always better than a wrong one.

★★★ CONSTRAINT 4: shot_transfer — Continuous Shot vs Hard Cut ★★★
Pay VERY careful attention to whether consecutive shots are truly separated by a hard cut
or are actually a continuous camera movement:

- **Hard cut**: A sudden, instantaneous change in the visual content — the scene, angle,
  or framing changes abruptly between two consecutive frames. There is a clear visual
  discontinuity.
  → Use: "Hard cut from [description of previous shot] to [description of new shot]."

- **Continuous shot / camera movement**: The camera smoothly pans, tilts, zooms, or
  tracks without any visual discontinuity. The transition is gradual.
  → Use: "Continuous shot with camera [panning/tilting/zooming] to [new framing]."
  → Or: "The camera continues [movement description], now showing [new content]."

- **How to tell the difference**: Look at the frames right at the shot boundary.
  If the visual content changes smoothly (same scene, gradual movement), it is CONTINUOUS.
  If there is an abrupt jump to a completely different framing/angle/scene, it is a HARD CUT.
  When in doubt, prefer "Continuous shot" over "Hard cut" — false hard cuts are worse
  than false continuity.

★★★ CONSTRAINT 5: Blurry/Rapid Motion Scenes ★★★
When the video contains rapid camera movement, motion blur, or shaky footage:
- Acknowledge the blur/shake honestly: "The camera rapidly pans, creating motion blur."
- Do NOT guess what is behind the blur — only describe what you can clearly see.
- If a segment is too blurry to identify specific actions, say so:
  → GOOD: "Rapid camera movement creates significant motion blur, obscuring the scene."
  → BAD: "[PERSON_1] walks across the room" (when the frame is too blurry to confirm)
- For very short blurry transitions (< 0.5s), describe them as transitional motion
  rather than trying to identify specific actions within them.
"""
    # Insert before the final instruction line
    insert_marker = "现在严格遵循以上所有核心原则"
    if insert_marker in prompt:
        idx = prompt.rfind(insert_marker)
        prompt = prompt[:idx] + _SHOT_EXTRA_CONSTRAINTS + "\n" + prompt[idx:]
    else:
        prompt += _SHOT_EXTRA_CONSTRAINTS

    return _inject_feedback(prompt, feedback)


# ======================================================================
# Fallback prompts (used when original files are inaccessible, maintaining full MTSS spec)
# Note: visual_description updated to array format [{"sub_shot_time": [...], "sub_shot_content": "..."}]
# ======================================================================

_FALLBACK_REF_PROMPT = """一、角色与任务
你现在是一位顶级的AIGC视频数据标注专家，完全精通并遵循一套名为 MultiTrack SceneScript 的精密视频描述规范。你的核心任务是接收一段输入视频，完成 MultiTrack SceneScript 的第一阶段：核心实体识别。你需要输出一个唯一、完整且格式绝对正确的 JSON 对象，仅包含 references（核心实体定义）。你的输出必须直接是 JSON 代码，不包含任何额外的解释性文字或代码块标记。

二、重要知识：
1、视频右下角的红色数字为当前视频帧在视频中的绝对时间点（单位为秒），一定要利用起来。
2、判断音频的时间点时，同样可以好好利用视频帧右下角标注的时间点信息。
3、注意识别视频是否是无声、静音视频。

三、核心规范
references 数组定义视频中所有关键实体：
[
    {
      "ref_id": "string",
      "type": "string",
      "semantic_description": "string",
      "appearance_anchor": {
         "id_features": {
            "detail_description": "string",
            "timestamp": "string"
        },
        "attributes": {
          "clothing": "string",
          "hairstyle": "string",
          "accessories": "string"
        }
      }
    }
]

ref_id 格式: PERSON_1, ANIMAL_1, OBJECT_1, SCENE_1 等。
同一人物多种装扮: PERSON_1_STYLE_1, PERSON_1_STYLE_2。
同一场景多视角: SCENE_1_VIEW_1, SCENE_1_VIEW_2。
type: person, animal, object, scene 之一。
当 type 不是 person 时，attributes 为 {}。
当 type 是 scene 时，timestamp 为 ""。

四、输出格式
输出仅包含 references 数组的 JSON 对象，所有描述内容以英文输出。
现在严格遵循以上所有核心原则并模仿范例，分析提供的视频，输出第一阶段的 JSON 描述（仅包含 references 数组），所有描述内容以英文输出。
CRITICAL: Output valid JSON only, no extra text, no markdown code blocks. ALL text content must be in English."""

_FALLBACK_EVENT_PROMPT = """一、角色与任务
你现在是一位顶级的AIGC视频数据标注专家。你的核心任务是接收一段输入视频和已生成的 references，完成第二阶段：音频事件时间线梳理。输出仅包含 events 数组的 JSON 对象。

二、events 数组规范
[
  {
    "event_id": "string",
    "type": "string",
    "time_range": [start_time, end_time],
    "content": {
      "speaker": "string",
      "line": "string",
      "description": "string"
    }
  }
]

event_id 格式: DIALOGUE_1, SFX_1, MUSIC_1。
type: dialogue, sfx, music 之一。
time_range: 绝对起止时间（秒，精确至0.1秒）。
speaker: 引用 ref_id；如果说话人未出现在画面中（旁白/画外音），则用文本描述声音来源（如 "an unseen male narrator"）。
line: 完整的、未被切分的连续对话原文，不做翻译。

★★★ ANTI-HALLUCINATION RULES FOR EVENTS ★★★
1. DIALOGUE events: Only create a DIALOGUE event if you can CLEARLY HEAR spoken words in the audio.
   - If the audio only contains music or ambient sounds, do NOT create any DIALOGUE events.
   - If you cannot clearly make out the words, do NOT guess — skip the event or use "[unclear speech]" as the line.
   - Verify speaker assignment: if the speaker is a visible person, they should be visibly speaking (mouth moving) at the specified time.
   - For narrators/voiceovers not visible in the video: use a descriptive speaker like "narrator", "male voiceover", "an unseen female narrator" instead of a ref_id.
2. SFX events: Only create SFX events for distinct, identifiable sound effects.
   - Do NOT create SFX events for normal ambient sounds that are part of the background.
   - Describe the sound accurately — do NOT confuse clapping with footsteps, etc.
3. MUSIC events: Only create MUSIC events if there is clearly identifiable music playing.
   - Do NOT fabricate song names, artists, or lyrics you cannot clearly hear.
4. If the video appears to be silent or only has background music, the events array should reflect that.
   It is perfectly acceptable to have an events array with only MUSIC events and no DIALOGUE.
5. It is MUCH BETTER to output fewer correct events than many hallucinated ones.

三、上一阶段输出的 references
{references}

四、输出格式
输出仅包含 events 数组的 JSON 对象，所有描述内容以英文输出。
现在严格遵循以上所有核心原则并模仿范例，结合提供的视频和上一阶段的 references 结果，分析并输出第二阶段的 JSON 描述（仅包含 events 数组），所有描述内容以英文输出。
IMPORTANT: content.line must preserve the ORIGINAL spoken language (do NOT translate). All other fields (description, etc.) must be in English.
CRITICAL: Output valid JSON only, no extra text, no markdown code blocks."""

_FALLBACK_SHOT_PROMPT = """一、角色与任务
你现在是一位顶级的AIGC视频数据标注专家。你的核心任务是接收一段输入视频和已生成的 references、events，完成第三阶段：镜头序列拆解与视觉描述。输出仅包含 shots 数组的 JSON 对象。

二、shots 数组规范
[
    {
      "shot_id": "string",
      "time_range": [start_time, end_time],
      "shot_transfer": "string",
      "visual_description": [
        {"sub_shot_time": [start, end], "sub_shot_content": "string"},
        ...
      ],
      "camera": {"shot_type": "string", "angle": "string", "movement": "string"},
      "active_events": ["EVENT_ID"],
      "references_in_shot": ["REF_ID"]
    }
]

shot_id 格式: shot_1, shot_2。
time_range: 绝对起止时间（秒，精确至0.1秒）。
shot_transfer: 一句话描述镜头是怎么切换过来的，包含转场方式和切换的视觉特征。
  - 例如: "Hard cut from the indoor office to an outdoor street scene."
  - 第一个镜头写 "Opening shot" 加上初始画面描述。
  - 不要在 shot_transfer 中详细描述当前镜头的场景环境（那些放在 visual_description 中）。
visual_description: 数组格式，镜头内按时序逐一记录每个最小粒度的原子动作。
  - 每条记录只描述一个原子动作（不可再拆分的最小视觉单元）。
  - sub_shot_content 必须使用 [ref_id] 指明动作主体。
  - 不同动作的 sub_shot_time 允许且鼓励交叉重叠。
  - 前几条记录用于描述场景环境、光影氛围、人物初始位置等静态信息（start 和 end 相同）。
  - 所有记录按 sub_shot_time 的 start 时间升序排列。

★★★ CRITICAL: sub_shot_time MUST NOT span the entire shot ★★★
  - FORBIDDEN: Using the full shot time_range (e.g., [5.96, 20.68]) as sub_shot_time for action descriptions.
    This means you are NOT breaking down the actions into atomic units.
  - EXCEPTION: Only the first 1-2 records (static scene description) may use the full shot time_range.
  - For dynamic actions (movements, gestures, expressions), sub_shot_time MUST be a SHORT time segment
    (typically 0.5s-3s) that precisely captures WHEN that specific action occurs.
  - Example of WRONG approach: sub_shot_time=[0.0, 15.0], content="[PERSON_1] speaks continuously"
  - Example of CORRECT approach: Break into multiple records:
    * [0.0, 2.5] "[PERSON_1] begins speaking with hands resting on the table"
    * [2.5, 4.0] "[PERSON_1] raises right hand to gesture"
    * [4.0, 6.0] "[PERSON_1] lowers hand back to table"
    * etc.

★★★ CRITICAL: Persons described in visual_description ★★★
  - If a person/entity described in visual_description is a MAIN character defined in references,
    you MUST use their [ref_id] and include them in references_in_shot.
  - If a person is a background/minor figure NOT defined in references, describe them with
    generic text (e.g., "An elderly man in a brown coat") WITHOUT using a ref_id.
    They do NOT need to be in references_in_shot.
active_events: 引用当前时间范围内的 event_id（基于时间重叠关系）。
references_in_shot: 列出画面中可见的所有 ref_id。

★★★ ANTI-HALLUCINATION — MANDATORY VERIFICATION PROTOCOL ★★★

Before writing EACH sub_shot_content entry, perform this 3-step verification:

**Step A: PERSON VERIFICATION** — For each [PERSON_X] you mention:
  - Can you clearly see this specific person at this exact time?
  - Can you distinguish them from other people? If not, do NOT use a ref_id.
  - Is this person actually performing the described action?

**Step B: ACTION VERIFICATION** — For each action you describe:
  - Can you clearly see this action happening in the video frames?
  - Is the action happening at the time you specified? Check the red timestamp overlay.
  - Are you describing what you SEE, not what you EXPECT?

**Step C: EXISTENCE VERIFICATION** — For each entity/object you mention:
  - Does this entity actually appear in the video at this time?
  - If a person exits the frame, do NOT describe them as still present.

Common hallucination patterns to AVOID:
1. "Screen goes dark/black": Only if you CLEARLY SEE a completely black screen.
2. "Wrong person attribution": The #1 most common error. VERIFY which person before writing.
3. "Fabricated dialogue": Do NOT create dialogue for music or ambient noise.
4. "Non-existent action": Do NOT describe actions you cannot clearly see.
5. "Person still present": When a person exits the frame, they are GONE.
6. Time-shifted descriptions: Cross-check with the red timestamp overlay.

GOLDEN RULE: When in doubt, OMIT. A missing description is always better than a wrong one.

★★★ shot_transfer — Continuous Shot vs Hard Cut ★★★
- Hard cut: Sudden, instantaneous change in visual content (abrupt scene/angle change).
  → Use: "Hard cut from [previous] to [new]."
- Continuous shot: Smooth camera pan/tilt/zoom without visual discontinuity.
  → Use: "Continuous shot with camera [movement] to [new framing]."
- When in doubt, prefer "Continuous shot" over "Hard cut".

★★★ Blurry/Rapid Motion Scenes ★★★
- Acknowledge blur/shake honestly. Do NOT guess what is behind the blur.
- GOOD: "Rapid camera movement creates significant motion blur."
- BAD: "[PERSON_1] walks across the room" (when frame is too blurry to confirm)

★ 最重要的约束：shots 数量和时间段必须严格遵循 shot_time ★

三、前两个阶段的输出结果
=== 第一阶段输出：references ===
{references}

=== 第二阶段输出：events ===
{events}

=== 预切分的镜头时间段（shot_time） ===
{shot_time}

四、输出格式
输出仅包含 shots 数组的 JSON 对象，所有描述内容以英文输出。
现在严格遵循以上所有核心原则并模仿范例，结合提供的视频以及前两个阶段的 references 和 events 结果，分析并输出第三阶段的 JSON 描述（仅包含 shots 数组），所有描述内容以英文输出。
CRITICAL: Output valid JSON only, no extra text, no markdown code blocks. ALL text content must be in English."""
