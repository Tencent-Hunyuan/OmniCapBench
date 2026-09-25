# MTSS Evaluation Metrics Documentation

This document describes the MTSS evaluation metrics as currently implemented in the code, corresponding to the following files:

- `mtss_metrics.py`: Rule-based metrics.
- `mtss_llm_judge.py`: Standalone LLM-as-Judge metrics.
- `../scripts/eval_all_metrics.sh`: Batch launch script responsible for conversion, rule-based metrics, and LLM judge.

The metric versions produced by the current code are:

- Rule-based metrics: `mtss_metrics_md_v1`
- LLM judge metrics: `mtss_llm_judge_metrics_md_v1`

The evaluation target of MTSS is not free-form text but a structured video state. The model must reconstruct references, events, shots, subshots, time ranges, and the linking relationships between these elements. Therefore, the evaluation pipeline first performs structural alignment, then separately assesses coverage, temporal localization, reference usage, event-shot linking, dialogue, and local visual states.

Rule-based metrics and LLM judge metrics are two independent tables. The current code does not compute an aggregate total score.

## Paper-Aligned Metric Names

The following table shows the exact metric names as they appear in the paper and their corresponding code output keys:

### Rule-Based Structural Metrics (Table 1 in paper)

| Paper Column | Code Key | Category | Description |
|---|---|---|---|
| SGC | `SGC` | Independent | Schema Gate Compliance |
| RefUse | `RefUse` | Visual | Reference ID usage correctness (mean of WHO_shot + WHO_sub) |
| CCC | `CCC` | Visual | Cross-Shot Coreference Consistency |
| Ref Subject F1 | `Ref_Subject_F1` | Visual | Subject reference detection (person+animal+object) |
| Ref Scene F1 | `Ref_Scene_F1` | Visual | Scene reference detection |
| Shot F1 | `shot_f1` | Visual | Shot boundary detection (tIoU >= 0.30) |
| Shot tIoU | `Shot_tIoU` / `WHEN_H` | Visual | Temporal precision for shots |
| Subshot F1 | `Subshot_F1` | Visual | Fine-grained visual state detection |
| Subshot tIoU | `Subshot_tIoU` / `WHEN_S` | Visual | Temporal precision for subshots |
| Event F1 | `event_f1` | Audio | Audio event detection (WER >= 0.50 for dialogue, tIoU >= 0.20 for non-dialogue) |
| Event tIoU | `Event_tIoU` / `WHEN_E` | Audio | Temporal alignment of audio events |
| Speaker F1 | `Speaker_F1` | Audio-Visual | Dialogue speaker identification accuracy |
| EVSA F1 | `EVSA` | Audio-Visual | Event-Shot Association |

### LLM-as-Judge Semantic Metrics (Table 2 in paper)

| Paper Column | Code Key | Category | Description |
|---|---|---|---|
| Shot | `LLM_shot_fact` | Visual | Shot visual description semantic quality |
| Subject Recall | `LLM_person_ref_recall` | Visual | Subject reference semantic recall |
| Subject Precision | `LLM_person_ref_precision` | Visual | Subject reference semantic precision |
| Scene Recall | `LLM_scene_ref_recall` | Visual | Scene reference semantic recall |
| Scene Precision | `LLM_scene_ref_precision` | Visual | Scene reference semantic precision |
| Subshot Recall | `LLM_subshot_recall` | Visual | Subshot semantic recall |
| Subshot Precision | `LLM_subshot_precision` | Visual | Subshot semantic precision |
| Dialogue | `DialogueLineScore` | Audio | Rule-based dialogue line structural score |
| Non-dialogue | `LLM_event` | Audio | Non-dialogue event semantic quality |

---

## 1. Data Interface

Each JSONL line corresponds to one video sample.

Ground truth is read from the following field:

```text
row.mtss_caption
```

Prediction results are read from the following field:

```text
row.model_prediction
```

Before evaluation, both are canonicalized into three lists:

```text
references
events
shots
```

The code handles a common malformed format: if `events` is mistakenly saved as a dictionary containing `references/events/shots`, the canonicalizer will move these three fields back to the top level. If the result still cannot produce three lists, the schema validity in SGC will be recorded as 0.

---

## 2. Reference Matching

The current `match_references` has been changed to LLM-assisted matching. The old pure-text ROUGE-L style detail matching is no longer used for reference alignment. To avoid breaking downstream calls, the function name, parameters, and return value remain unchanged:

```python
match_references(gt_refs, pred_refs, threshold=REF_THRESHOLD) -> Dict[str, str]
```

The return value is still:

```text
prediction_ref_id -> gt_ref_id
```

Only successfully matched prediction references appear in this dictionary. Unmatched items are not written to the dictionary, so the calling convention for downstream code remains the same as before.

### 2.1 Grouping by Type

References are first grouped by type. The current code only performs LLM matching for these three types:

```text
person
scene
object
```

For each type, all GT references and all prediction references of that type are sent to the LLM together. Each item includes:

```text
index
ref_id
detail_description
```

Here, `detail_description` refers to:

```text
appearance_anchor.id_features.detail_description
```

Reference matching only uses this detail description; it does not use `semantic_description`, attributes, timestamp, event, shot, or other contextual fields.

### 2.2 LLM One-to-One Matching Rules

For each type group, the LLM sees the complete GT list and prediction list for that type simultaneously and performs one-to-one matching:

- A prediction reference can match at most one GT reference.
- A GT reference can be used by at most one prediction reference.
- A successful match is recorded as `match=1`.
- A failed match is recorded as `match=0`.
- Cases of vague similarity, empty descriptions, insufficient information, or contradictions in core identity/appearance/scene/object details should all be judged as non-matching.

The expected LLM output format is:

```json
{
  "matches": [
    {"pred_ref_id": "PERSON_1", "gt_ref_id": "PERSON_1", "match": 1},
    {"pred_ref_id": "PERSON_2", "gt_ref_id": "", "match": 0}
  ]
}
```

A defensive check is performed during parsing. The code accepts `{"matches": [...]}`, a bare JSON list, or a simple `{pred_id: gt_id}` dictionary. After parsing, it verifies that IDs belong to the current type group and enforces one-to-one matching again. If the LLM returns duplicate matches (e.g., two predictions both pointing to the same GT), only the first valid and unoccupied pair is kept; subsequent duplicate pairs are ignored. This ensures that even imperfect LLM output will not produce one-to-many matches.

### 2.3 LLM Configuration for Reference Matching

Reference matching uses `UnifiedTextClient` from `mtss_api_clients.py`. It uses the standard OpenAI-compatible chat completions API and supports any compatible endpoint:

```text
- OpenAI (https://api.openai.com/v1)
- OpenRouter (https://openrouter.ai/api/v1)
- DashScope / Qwen (https://dashscope.aliyuncs.com/compatible-mode/v1)
- Self-hosted vLLM / TGI (http://localhost:8000/v1)
```

The matcher is configured through this environment variable group:

```text
REF_MATCH_BASE_URL       OpenAI-compatible endpoint (default http://localhost:8000/v1)
REF_MATCH_MODEL          matcher model name        (default gpt-4o)
REF_MATCH_API_KEY        API key                   (default EMPTY)
REF_MATCH_TEMPERATURE    sampling temperature      (default 0.0)
REF_MATCH_MAX_TOKENS     max output tokens         (default 4096)
REF_MATCH_TIMEOUT        per-request timeout       (default 600)
REF_MATCH_MAX_RETRIES    client-level retries      (default 5)
```

If neither `REF_MATCH_BASE_URL` nor `REF_MATCH_API_KEY` is set, LLM-assisted matching is skipped entirely and `match_references` falls back to exact reference-ID matching. This keeps the rule-based metrics runnable fully offline, but `Ref`, `RefUse`, and `CCC` will then only credit predictions that reuse the ground-truth identifiers verbatim.

When `mtss_llm_judge.py` creates the `LLMJudgeClient`, it syncs the current judge configuration into `REF_MATCH_*`, so reference matching during judge task collection is consistent with the judge model.

`scripts/eval_all_metrics.sh` passes the selected judge endpoint and model to the rule-based metrics through the same `REF_MATCH_*` group. This matters because `Ref`, `RefUse`, and `CCC` depend on the LLM-generated reference map.

---

## 3. Rule-Based Metrics

Rule-based metrics produce a summary JSON and a per-video detail JSONL. The main table contains `SGC` plus 8 compact metric families:

```text
SGC
Ref
RefUse
CCC
WHEN
EVSA
Event
Dialog
Subshot
```

The per-video detail also saves expanded fields including recall, precision, F1, match counts, and diagnostic fields.

### 3.1 SGC

`SGC` is a structural completeness gate that does not judge factual correctness. It only answers one question: is the model output complete enough to be scored as an MTSS state.

SGC averages the following components:

| Component | What It Checks |
|---|---|
| `sgc_schema_valid` | Whether `references`, `events`, `shots` can be canonicalized into three lists |
| `sgc_ref_completeness` | Whether prediction references contain `ref_id`, `type`, `semantic_description`, and `detail_description` |
| `sgc_event_completeness` | Whether prediction events contain `event_id`, `type`, valid `time_range`, and `content.description`; dialogue events additionally require `content.line` |
| `sgc_shot_completeness` | Whether prediction shots contain `shot_id`, valid `time_range`, `shot_transfer`, `visual_description`, and `camera` |
| `sgc_subshot_completeness` | Whether prediction subshots contain valid `sub_shot_time` and non-empty `sub_shot_content` |
| `sgc_link_field_completeness` | Whether prediction shots expose both `active_events` and `references_in_shot` lists |
| `sgc_count_coverage` | Whether the prediction's references/events/shots counts relative to GT are not obviously empty |

### 3.2 Ref

`Ref` measures whether GT references are recalled and whether prediction references can be matched back to GT references.

```text
ref_recall    = num_matched_refs / num_gt_refs
ref_precision = num_matched_refs / num_pred_refs
Ref           = mean(ref_recall, ref_precision)
```

`num_matched_refs` is the number of pairs returned by the LLM-assisted one-to-one reference matcher.

### 3.3 Shot Matching, Shot Coverage, and WHEN_H

Shots are matched by temporal IoU. A shot pair becomes a candidate when:

```text
IoU(pred.time_range, gt.time_range) >= SHOT_IOU_THRESHOLD
```

The current threshold is:

```text
SHOT_IOU_THRESHOLD = 0.30
```

Candidate pairs are greedily matched from highest to lowest IoU, with one-to-one enforcement.

Matched shot pairs are used to compute shot-level coverage:

```text
shot_recall    = num_matched_shots / num_gt_shots
shot_precision = num_matched_shots / num_pred_shots
shot_f1        = F1(shot_precision, shot_recall)
```

These three fields describe how many shots are aligned. They are written to the JSON summary and detail output as expanded metrics; they currently do not appear as separate compact columns in the main table.

`WHEN_H` measures the temporal quality of successfully matched shot pairs, i.e., the average temporal IoU of matched shot pairs. It differs from `shot_f1` in meaning: `shot_f1` measures coverage, while `WHEN_H` measures the degree of temporal overlap for already-matched pairs.

### 3.4 Event Matching, Event, Dialog, and WHEN_E

Event matching is divided into dialogue and non-dialogue categories.

Dialogue events:

- Both GT and prediction `type` must be `dialogue`.
- The matching score is the normalized WER similarity of `content.line`.
- The current threshold is `DIALOGUE_LINE_THRESHOLD = 0.50`.

Non-dialogue events:

- Both GT and prediction must not be `dialogue`.
- The matching score is the temporal IoU of the event `time_range`.
- The current threshold is `EVENT_TIME_THRESHOLD = 0.20`.

Candidate event pairs are also greedily matched from highest to lowest score, with one-to-one enforcement.

The Event compact metric is:

```text
event_recall    = num_matched_events / num_gt_events
event_precision = num_matched_events / num_pred_events
event_f1        = F1(event_precision, event_recall)
Event           = event_f1
```

`WHEN_E` is the average temporal IoU of all matched event pairs, including dialogue pairs.

Dialogue-related metrics are:

```text
DialogueLineScore   = average normalized WER score of matched dialogue lines
WHAT_dialogue_sem   = DialogueLineScore
DialogueMergedScore = normalized WER score after temporally concatenating all dialogue
Dialog              = DialogueLineScore
```

The current code does not send dialogue to the LLM judge.

### 3.5 RefUse

`RefUse` checks whether references are correctly used in matched shots, matched events, and matched subshots. Prediction reference IDs are first mapped to the GT ID space via the LLM reference map.

The expanded fields are:

```text
WHO_shot
WHO_event
WHO_sub
```

For matched shots, `WHO_shot` is the average Jaccard overlap between mapped prediction `references_in_shot` and GT `references_in_shot`.

For matched events, `WHO_event` is the average Jaccard overlap between mapped prediction event references and GT event references. Event references are extracted from event content and common fields such as `speaker`, `source_ref`, `target_ref`, and `ref_id`.

For matched subshots, `WHO_sub` checks whether the matched prediction subshot group covers the references required by the GT subshot.

The compact score is:

```text
RefUse = mean(WHO_shot, WHO_event, WHO_sub)
```

Only applicable components are included in the mean.

### 3.6 CCC

`CCC` measures whether each GT reference appears on the correct subshot temporal support.

For each GT reference, the code collects:

- All subshot time segments in GT where `sub_shot_content` mentions that GT ref ID.
- All subshot time segments in prediction where prediction ref IDs mapped to that GT ref appear in `sub_shot_content`.

Then it computes the union IoU of the two sets of time segments.

Special cases:

- If both GT and prediction sides have empty temporal support, that reference is scored as 1.
- If only one side is empty, that reference is scored as 0.

The reported field is:

```text
CCC = CCC_macro = mean of reference-level support IoU across GT references
```

### 3.7 Subshot Matching, Subshot, and WHEN_S

Rule-based subshot matching uses a local temporal window.

For each matched shot pair, the code iterates over GT subshots within the GT shot. A prediction subshot enters the candidate group if it overlaps with the following extended window around the GT subshot:

```text
[gt_start - SUBSHOT_WINDOW_DELTA, gt_end + SUBSHOT_WINDOW_DELTA]
```

The current value is:

```text
SUBSHOT_WINDOW_DELTA = 1.0
```

If the candidate group is non-empty, that GT subshot is counted as matched. All prediction subshots that enter any candidate group are counted as matched predictions.

The expanded metrics are:

```text
Subshot_Recall    = num_matched_gt_subshots / num_gt_subshots
Subshot_Precision = num_matched_pred_subshots / num_pred_subshots
Subshot_F1        = F1(Subshot_Precision, Subshot_Recall)
Subshot           = Subshot_F1
```

`WHEN_S` is the average union IoU between matched GT subshot time segments and the temporal union of their matched prediction subshot group.

### 3.8 WHEN

The compact temporal metric is:

```text
WHEN = mean(WHEN_H, WHEN_E, WHEN_S)
```

Only applicable components are included in the mean.

### 3.9 EVSA

`EVSA` measures whether event-shot links are correctly reconstructed. It first maps prediction events and prediction shots to the GT ID space, then compares event-shot edges.

GT edges come from:

```text
gt_shot.active_events
```

Prediction edges come from:

```text
pred_shot.active_events
```

Only prediction edges where both the event and shot are successfully matched to GT can participate in the computation.

The expanded metrics are:

```text
EVSA_P = correct_mapped_pred_edges / mapped_pred_edges
EVSA_R = correct_mapped_pred_edges / gt_edges
EVSA   = F1(EVSA_P, EVSA_R)
```

The main table `EVSA` uses F1.

---

## 4. LLM-as-Judge Metrics

`mtss_llm_judge.py` is a standalone evaluator that does not modify the rule-based metrics table and does not output an aggregate score.

It first canonicalizes GT and prediction, then reuses the alignment utilities from `mtss_metrics.py`:

```text
match_references
match_events
match_shots
```

Since `match_references` is now LLM-assisted matching, the LLM judge also uses LLM-assisted reference matching during task collection.

### 4.1 Reference Quality

The reference quality judge targets matched person and matched scene references. Each matched pair is scored twice:

```text
LLM_person_ref_recall
LLM_person_ref_precision
LLM_scene_ref_recall
LLM_scene_ref_precision
```

**Recall direction**: Using the GT `detail_description` as the subject, checks whether the prediction's description covers all factual content in the GT. Scoring focuses on whether each specific factual element in the GT (e.g., clothing color, hairstyle, body type, scene layout, lighting conditions) is mentioned in the prediction. Omissions are penalized; contradictions are penalized more heavily.

**Precision direction**: Using the prediction's `detail_description` as the subject, checks whether the GT's description covers all factual content in the prediction. Facts present in the prediction but absent from the GT are considered potential hallucinations/unsupported claims. Contradictions are more severe than unsupported claims.

Each direction only compares:

```text
appearance_anchor.id_features.detail_description
```

The LLM outputs a score from 1 to 5. During aggregation, scores are normalized to `[0, 1]`:

```text
normalized = (score - 1) / 4
```

The compact fields are the mean of recall and precision:

```text
LLM_person_ref = mean(LLM_person_ref_recall, LLM_person_ref_precision)
LLM_scene_ref  = mean(LLM_scene_ref_recall, LLM_scene_ref_precision)
```

Object references participate in the reference map and affect downstream rule-based metrics and some context, but the current LLM judge table does not have a separate `LLM_object_ref` field.

### 4.2 Event Quality

`LLM_event` only targets matched non-dialogue events.

It compares:

```text
content.description
```

Scoring focuses on whether all factual elements in the GT description are covered by the prediction, including: main actions/activities, involved entities and their roles, spatial relationships between entities, causal or sequential sub-actions, environment/audio context related to the event, and results or state changes. Each factual element is an independent coverage point.

Contradictions (stating the opposite) are more severe than omissions. Extra details in the prediction not mentioned in the GT neither penalize nor reward.

It does not score dialogue lines, event type accuracy, event timing, or shot placement. The reference mapping is provided to the LLM as context to help it understand IDs in event text.

### 4.3 Subshot LLM Binary Classification Matching

LLM subshot matching is a local binary classification task, separate from the 1-to-5 subshot semantic quality scoring.

The two binary classification dimensions are:

```text
LLM_subshot_match_precision
LLM_subshot_match_recall
```

Prediction-side precision direction: Each predicted subshot serves as the target. The code finds its containing matched GT shot and projects a small GT candidate group based on subshot index, then asks the LLM whether that target prediction has at least one true counterpart.

GT-side recall direction: Each GT subshot serves as the target. The code finds its containing matched predicted shot and projects a small prediction candidate group based on subshot index, then asks the LLM whether that target GT has at least one true counterpart.

The candidate group size is controlled by two radius parameters:

```text
SHOT_CONTEXT_RADIUS = 2
SUBSHOT_CONTEXT_RADIUS = 2
```

`SHOT_CONTEXT_RADIUS` determines how many shots to expand forward and backward from the matched shot in the opposite-side shot sequence to collect candidate subshots. `SUBSHOT_CONTEXT_RADIUS` determines how many subshots to take forward and backward from the projected position within those candidates as the final candidate group.

The LLM prompt requires output of only:

```text
<match>1</match>
```

or:

```text
<match>0</match>
```

This LLM subshot matching is not one-to-one matching. It determines whether each target has at least one corresponding state in the local candidate group. During aggregation, matched GT keys and matched prediction keys are deduplicated.

The output summary saves:

```text
llm_subshot_local_matching.Subshot_Recall
llm_subshot_local_matching.Subshot_Precision
llm_subshot_local_matching.Subshot_F1
llm_subshot_local_matching.num_gt_subshots
llm_subshot_local_matching.num_pred_subshots
llm_subshot_local_matching.num_matched_gt_subshots
llm_subshot_local_matching.num_matched_pred_subshots
```

### 4.4 Subshot Semantic Quality

The 1-to-5 subshot semantic quality scoring is gated by the binary classification match result.

Only when the target's `match=1` are the following further computed:

```text
LLM_subshot_precision
LLM_subshot_recall
```

The factual elements evaluated include: which entities are visible (persons/objects with ref IDs), current posture/action/gesture/expression, spatial positions and relationships between entities, interactions with objects or the environment, direction/speed of movement, and any transient visual states (e.g., lighting changes, objects appearing/disappearing). Each factual element is an independent coverage/verification point.

Precision direction: Using the prediction subshot as the target, checks whether its facts are supported by the GT context. Prediction details not mentioned in the GT are not penalized; only those contradicting the GT are penalized.

Recall direction: Using the GT subshot as the target, checks whether the prediction context covers all facts in the GT. Both omissions and contradictions are penalized, with contradictions being more severe.

The compact field:

```text
LLM_subshot
```

is the mean of precision-side and recall-side semantic scores after combining raw 1-5 scores and normalizing uniformly:

```text
LLM_subshot = (mean(all_precision_scores + all_recall_scores) - 1) / 4
```

### 4.5 Shot Fact Quality

`LLM_shot_fact` targets matched shot pairs and only creates tasks when comparable shot fact fields exist.

It only scores:

```text
camera
shot_transfer
```

Scoring focuses on whether all camera/transfer facts in the GT are covered by the prediction, including: shot scale (close-up/medium/full/long), camera angle (high/low/eye-level/bird's-eye/Dutch angle), camera movement (pan/tilt/push-in/pull-out/tracking/crane/handheld/static), composition (rule of thirds/centered/offset), depth of field, focal subject, transition type (hard cut/dissolve/fade in-out/wipe/match cut), and transition speed.

Contradictions are more severe than omissions. Extra details in the prediction are not penalized.

It does not score shot visual description, active events, references in shot, event content, or timing.

---

## 5. Batch Script Behavior

`scripts/eval_all_metrics.sh` supports:

```text
all
rule-base
rule-based
llm-only
convert-only
```

The judge endpoint is selected with these variables, all of which have in-script defaults:

```text
JUDGE_BASE_URL      OpenAI-compatible endpoint (default http://localhost:8000/v1)
JUDGE_MODEL         judge model name           (default gpt-4o)
JUDGE_API_KEY       API key                    (default EMPTY)
JUDGE_WORKERS       parallel judge requests    (default 8)
JUDGE_TEMPERATURE   sampling temperature       (default 0.7)
JUDGE_MAX_TOKENS    max output tokens          (default 4096)
METRICS_WORKERS     parallel video scoring     (default 4)
```

Any OpenAI-compatible provider works — OpenAI, OpenRouter, DashScope, or a self-hosted vLLM server.

The selected endpoint simultaneously controls:

- LLM judge calls in `mtss_llm_judge.py`, passed as command-line arguments.
- LLM-assisted reference matching in `mtss_metrics.py`, passed as the `REF_MATCH_*` group.

When running LLM-only metrics, `mtss_llm_judge.py` sets the same `REF_MATCH_*` group internally so that reference matching during task collection uses the judge model.

Transformer captioner baselines, if needed, are first converted to MTSS format via `mtss_transformer_convert.py`. The conversion output is written to:

```text
results_mtss_converted.jsonl
```

Non-transformer models directly use:

```text
results.jsonl
```

---

## 6. Output Fields

The rule-based metrics main table reads:

```text
summary.main_table.SGC
summary.main_table.Ref
summary.main_table.RefUse
summary.main_table.CCC
summary.main_table.WHEN
summary.main_table.EVSA
summary.main_table.Event
summary.main_table.Dialog
summary.main_table.Subshot
```

The rule-based metrics detail file contains the following expanded fields:

```text
ref_recall
ref_precision
WHO_shot
WHO_event
WHO_sub
CCC_macro
WHEN_H
WHEN_E
WHEN_S
shot_recall
shot_precision
shot_f1
EVSA_P
EVSA_R
event_recall
event_precision
event_f1
DialogueLineScore
WHAT_dialogue_sem
DialogueMergedScore
Subshot_Recall
Subshot_Precision
Subshot_F1
num_gt_refs
num_pred_refs
num_matched_refs
num_gt_events
num_pred_events
num_matched_events
num_gt_shots
num_pred_shots
num_matched_shots
num_gt_subshots
num_pred_subshots
num_matched_gt_subshots
num_matched_pred_subshots
```

The LLM judge table reports:

```text
LLM_person_ref
LLM_person_ref_recall
LLM_person_ref_precision
LLM_scene_ref
LLM_scene_ref_recall
LLM_scene_ref_precision
LLM_event
LLM_subshot
LLM_subshot_precision
LLM_subshot_recall
LLM_shot_fact
```

The LLM subshot local matching block separately reports the recall, precision, and F1 obtained from binary classification matching. This is distinct from the 1-to-5 subshot semantic quality scores — they represent two different sets of information.

---

## 7. Currently Unreported Metrics

The current code intentionally does not report:

- Aggregate total score.
- Standalone event type accuracy.
- Standalone shot boundary error diagnostic.
- Prediction-side hallucination or unsupported-extra rate.
- Dialogue LLM judge score.
- `LLM_object_ref`.

These choices are made to keep the main table compact and ensure that each number directly corresponds to a well-defined component of the MTSS state.
