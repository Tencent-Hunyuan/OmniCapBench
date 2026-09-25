# MTSS 评估指标文档

本文档描述了当前代码中实现的 MTSS 评估指标，对应以下文件：

- `mtss_metrics.py`：基于规则的指标。
- `mtss_llm_judge.py`：独立的 LLM-as-Judge 指标。
- `../scripts/eval_all_metrics.sh`：批处理启动脚本，负责格式转换、规则指标和 LLM 评判。

当前代码产出的指标版本为：

- 规则指标：`mtss_metrics_md_v1`
- LLM 评判指标：`mtss_llm_judge_metrics_md_v1`

MTSS 的评估目标不是自由文本，而是结构化的视频状态。模型必须重建引用（references）、事件（events）、镜头（shots）、子镜头（subshots）、时间范围以及这些元素之间的关联关系。因此，评估流程首先进行结构对齐，然后分别评估覆盖率、时间定位、引用使用、事件-镜头关联、对话和局部视觉状态。

规则指标和 LLM 评判指标是两张独立的表格。当前代码不计算聚合总分。

## 论文对齐指标名称

下表展示了论文中出现的指标名称与代码输出键的精确对应关系：

### 基于规则的结构指标（论文表1）

| 论文列名 | 代码键 | 类别 | 描述 |
|---|---|---|---|
| SGC | `SGC` | 独立 | 模式门控合规性 |
| RefUse | `RefUse` | 视觉 | 引用ID使用正确性（WHO_shot + WHO_sub 的均值） |
| CCC | `CCC` | 视觉 | 跨镜头共指一致性 |
| Ref Subject F1 | `Ref_Subject_F1` | 视觉 | 主体引用检测（人物+动物+物体） |
| Ref Scene F1 | `Ref_Scene_F1` | 视觉 | 场景引用检测 |
| Shot F1 | `shot_f1` | 视觉 | 镜头边界检测（tIoU >= 0.30） |
| Shot tIoU | `Shot_tIoU` / `WHEN_H` | 视觉 | 镜头时间精度 |
| Subshot F1 | `Subshot_F1` | 视觉 | 细粒度视觉状态检测 |
| Subshot tIoU | `Subshot_tIoU` / `WHEN_S` | 视觉 | 子镜头时间精度 |
| Event F1 | `event_f1` | 音频 | 音频事件检测（对话WER >= 0.50，非对话tIoU >= 0.20） |
| Event tIoU | `Event_tIoU` / `WHEN_E` | 音频 | 音频事件时间对齐 |
| Speaker F1 | `Speaker_F1` | 音视觉 | 对话说话人识别准确率 |
| EVSA F1 | `EVSA` | 音视觉 | 事件-镜头关联 |

### LLM-as-Judge 语义指标（论文表2）

| 论文列名 | 代码键 | 类别 | 描述 |
|---|---|---|---|
| Shot | `LLM_shot_fact` | 视觉 | 镜头视觉描述语义质量 |
| Subject Recall | `LLM_person_ref_recall` | 视觉 | 主体引用语义召回率 |
| Subject Precision | `LLM_person_ref_precision` | 视觉 | 主体引用语义精确率 |
| Scene Recall | `LLM_scene_ref_recall` | 视觉 | 场景引用语义召回率 |
| Scene Precision | `LLM_scene_ref_precision` | 视觉 | 场景引用语义精确率 |
| Subshot Recall | `LLM_subshot_recall` | 视觉 | 子镜头语义召回率 |
| Subshot Precision | `LLM_subshot_precision` | 视觉 | 子镜头语义精确率 |
| Dialogue | `DialogueLineScore` | 音频 | 基于规则的对话行结构分数 |
| Non-dialogue | `LLM_event` | 音频 | 非对话事件语义质量 |

---

## 1. 数据接口

每行 JSONL 对应一个视频样本。

Ground truth 从以下字段读取：

```text
row.mtss_caption
```

预测结果从以下字段读取：

```text
row.model_prediction
```

评估前，两者均被规范化为三个列表：

```text
references
events
shots
```

代码处理一种常见的格式错误：如果 `events` 被错误保存为包含 `references/events/shots` 的字典，规范化器会将这三个字段移回顶层。如果结果仍无法产生三个列表，SGC 中的模式有效性将记录为 0。

---

## 2. 引用匹配

当前 `match_references` 已改为 LLM 辅助匹配。旧的纯文本 ROUGE-L 风格细节匹配不再用于引用对齐。为避免破坏下游调用，函数名、参数和返回值保持不变：

```python
match_references(gt_refs, pred_refs, threshold=REF_THRESHOLD) -> Dict[str, str]
```

返回值仍为：

```text
prediction_ref_id -> gt_ref_id
```

只有成功匹配的预测引用才会出现在该字典中。未匹配项不写入字典，因此下游代码的调用约定与之前相同。

### 2.1 按类型分组

引用首先按类型分组。当前代码仅对以下三种类型执行 LLM 匹配：

```text
person
scene
object
```

对于每种类型，所有该类型的 GT 引用和所有该类型的预测引用会一起发送给 LLM。每个条目包含：

```text
index
ref_id
detail_description
```

此处 `detail_description` 指的是：

```text
appearance_anchor.id_features.detail_description
```

引用匹配仅使用此细节描述；不使用 `semantic_description`、属性、时间戳、事件、镜头或其他上下文字段。

### 2.2 LLM 一对一匹配规则

对于每个类型组，LLM 同时看到该类型的完整 GT 列表和预测列表，并执行一对一匹配：

- 一个预测引用最多匹配一个 GT 引用。
- 一个 GT 引用最多被一个预测引用使用。
- 成功匹配记录为 `match=1`。
- 失败匹配记录为 `match=0`。
- 模糊相似、空描述、信息不足或核心身份/外观/场景/物体细节矛盾的情况均应判定为不匹配。

LLM 预期输出格式为：

```json
{
  "matches": [
    {"pred_ref_id": "PERSON_1", "gt_ref_id": "PERSON_1", "match": 1},
    {"pred_ref_id": "PERSON_2", "gt_ref_id": "", "match": 0}
  ]
}
```

解析时执行防御性检查。代码接受 `{"matches": [...]}`、裸 JSON 列表或简单的 `{pred_id: gt_id}` 字典。解析后验证 ID 属于当前类型组，并再次强制一对一匹配。如果 LLM 返回重复匹配（如两个预测都指向同一 GT），仅保留第一个有效且未被占用的配对；后续重复配对被忽略。这确保即使 LLM 输出不完美也不会产生一对多匹配。

### 2.3 引用匹配的 LLM 配置

引用匹配使用 `mtss_api_clients.py` 中的 `UnifiedTextClient`。它使用标准 OpenAI 兼容的 chat completions API，支持任何兼容端点：

```text
- OpenAI (https://api.openai.com/v1)
- OpenRouter (https://openrouter.ai/api/v1)
- DashScope / Qwen (https://dashscope.aliyuncs.com/compatible-mode/v1)
- 自托管 vLLM / TGI (http://localhost:8000/v1)
```

匹配器通过以下环境变量组配置：

```text
REF_MATCH_BASE_URL       OpenAI 兼容端点（默认 http://localhost:8000/v1）
REF_MATCH_MODEL          匹配器模型名       （默认 gpt-4o）
REF_MATCH_API_KEY        API key            （默认 EMPTY）
REF_MATCH_TEMPERATURE    采样温度           （默认 0.0）
REF_MATCH_MAX_TOKENS     最大输出 token     （默认 4096）
REF_MATCH_TIMEOUT        单次请求超时       （默认 600）
REF_MATCH_MAX_RETRIES    客户端级重试次数   （默认 5）
```

如果 `REF_MATCH_BASE_URL` 与 `REF_MATCH_API_KEY` 均未设置，则完全跳过 LLM 辅助匹配，`match_references` 回退为引用 ID 精确匹配。这使规则指标可以完全离线运行，但此时 `Ref`、`RefUse`、`CCC` 只会认可逐字复用了 GT 标识符的预测。

当 `mtss_llm_judge.py` 创建 `LLMJudgeClient` 时，会将当前评判配置同步到 `REF_MATCH_*`，因此评判任务收集期间的引用匹配与评判模型保持一致。

`scripts/eval_all_metrics.sh` 通过同一个 `REF_MATCH_*` 组把选定的评判端点和模型传给规则指标。这很重要，因为 `Ref`、`RefUse` 和 `CCC` 依赖于 LLM 生成的引用映射。

---

## 3. 规则指标

规则指标产出一个摘要 JSON 和一个逐视频详情 JSONL。主表包含 `SGC` 加 8 个紧凑指标族：

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

逐视频详情还保存了扩展字段，包括召回率、精确率、F1、匹配计数和诊断字段。

### 3.1 SGC

`SGC` 是一个结构完整性门控，不评判事实正确性。它只回答一个问题：模型输出是否足够完整以作为 MTSS 状态进行评分。

SGC 对以下组件取平均：

| 组件 | 检查内容 |
|---|---|
| `sgc_schema_valid` | `references`、`events`、`shots` 能否规范化为三个列表 |
| `sgc_ref_completeness` | 预测引用是否包含 `ref_id`、`type`、`semantic_description` 和 `detail_description` |
| `sgc_event_completeness` | 预测事件是否包含 `event_id`、`type`、有效 `time_range` 和 `content.description`；对话事件还需要 `content.line` |
| `sgc_shot_completeness` | 预测镜头是否包含 `shot_id`、有效 `time_range`、`shot_transfer`、`visual_description` 和 `camera` |
| `sgc_subshot_completeness` | 预测子镜头是否包含有效 `sub_shot_time` 和非空 `sub_shot_content` |
| `sgc_link_field_completeness` | 预测镜头是否暴露 `active_events` 和 `references_in_shot` 列表 |
| `sgc_count_coverage` | 预测的 references/events/shots 计数相对于 GT 是否不明显为空 |

### 3.2 Ref

`Ref` 衡量 GT 引用是否被召回，以及预测引用是否能匹配回 GT 引用。

```text
ref_recall    = num_matched_refs / num_gt_refs
ref_precision = num_matched_refs / num_pred_refs
Ref           = mean(ref_recall, ref_precision)
```

`num_matched_refs` 是 LLM 辅助一对一引用匹配器返回的配对数。

### 3.3 镜头匹配、镜头覆盖和 WHEN_H

镜头通过时间 IoU 进行匹配。当满足以下条件时，一对镜头成为候选配对：

```text
IoU(pred.time_range, gt.time_range) >= SHOT_IOU_THRESHOLD
```

当前阈值为：

```text
SHOT_IOU_THRESHOLD = 0.30
```

候选配对按 IoU 从高到低贪心匹配，强制一对一。

匹配的镜头配对用于计算镜头级覆盖率：

```text
shot_recall    = num_matched_shots / num_gt_shots
shot_precision = num_matched_shots / num_pred_shots
shot_f1        = F1(shot_precision, shot_recall)
```

这三个字段描述有多少镜头被对齐。它们被写入 JSON 摘要和详情输出作为扩展指标；目前不作为主表中的独立紧凑列出现。

`WHEN_H` 衡量成功匹配的镜头配对的时间质量，即匹配镜头配对的平均时间 IoU。它与 `shot_f1` 的含义不同：`shot_f1` 衡量覆盖率，而 `WHEN_H` 衡量已匹配配对的时间重叠程度。

### 3.4 事件匹配、Event、Dialog 和 WHEN_E

事件匹配分为对话和非对话两类。

对话事件：

- GT 和预测的 `type` 均须为 `dialogue`。
- 匹配分数为 `content.line` 的归一化 WER 相似度。
- 当前阈值为 `DIALOGUE_LINE_THRESHOLD = 0.50`。

非对话事件：

- GT 和预测均不为 `dialogue`。
- 匹配分数为事件 `time_range` 的时间 IoU。
- 当前阈值为 `EVENT_TIME_THRESHOLD = 0.20`。

候选事件配对同样按分数从高到低贪心匹配，强制一对一。

Event 紧凑指标为：

```text
event_recall    = num_matched_events / num_gt_events
event_precision = num_matched_events / num_pred_events
event_f1        = F1(event_precision, event_recall)
Event           = event_f1
```

`WHEN_E` 是所有匹配事件配对的平均时间 IoU，包括对话配对。

对话相关指标为：

```text
DialogueLineScore   = 匹配对话行的平均归一化WER分数
WHAT_dialogue_sem   = DialogueLineScore
DialogueMergedScore = 时间拼接所有对话后的归一化WER分数
Dialog              = DialogueLineScore
```

当前代码不将对话发送给 LLM 评判。

### 3.5 RefUse

`RefUse` 检查引用是否在匹配的镜头、匹配的事件和匹配的子镜头中被正确使用。预测引用 ID 首先通过 LLM 引用映射转换到 GT ID 空间。

扩展字段为：

```text
WHO_shot
WHO_event
WHO_sub
```

对于匹配的镜头，`WHO_shot` 是映射后的预测 `references_in_shot` 与 GT `references_in_shot` 之间的平均 Jaccard 重叠。

对于匹配的事件，`WHO_event` 是映射后的预测事件引用与 GT 事件引用之间的平均 Jaccard 重叠。事件引用从事件内容和通用字段如 `speaker`、`source_ref`、`target_ref`、`ref_id` 中提取。

对于匹配的子镜头，`WHO_sub` 检查匹配的预测子镜头组是否覆盖了 GT 子镜头所需的引用。

紧凑分数为：

```text
RefUse = mean(WHO_shot, WHO_event, WHO_sub)
```

仅适用的组件被纳入均值计算。

### 3.6 CCC

`CCC` 衡量每个 GT 引用是否出现在正确的子镜头时间支撑上。

对于每个 GT 引用，代码收集：

- GT 中所有 `sub_shot_content` 提及该 GT ref ID 的子镜头时间段。
- 预测中所有映射到该 GT ref 的预测 ref ID 出现在 `sub_shot_content` 中的子镜头时间段。

然后计算两组时间段的联合 IoU。

特殊情况：

- 如果 GT 和预测两侧的时间支撑均为空，该引用得分为 1。
- 如果只有一侧为空，该引用得分为 0。

报告的字段为：

```text
CCC = CCC_macro = GT 引用上引用级支撑 IoU 的均值
```

### 3.7 子镜头匹配、Subshot 和 WHEN_S

规则子镜头匹配使用局部时间窗口。

对于每对匹配的镜头配对，代码遍历 GT 镜头内的 GT 子镜头。如果一个预测子镜头与以下围绕 GT 子镜头的扩展窗口重叠，则进入候选组：

```text
[gt_start - SUBSHOT_WINDOW_DELTA, gt_end + SUBSHOT_WINDOW_DELTA]
```

当前值为：

```text
SUBSHOT_WINDOW_DELTA = 1.0
```

如果候选组非空，该 GT 子镜头被计为已匹配。所有进入任何候选组的预测子镜头被计为已匹配预测。

扩展指标为：

```text
Subshot_Recall    = num_matched_gt_subshots / num_gt_subshots
Subshot_Precision = num_matched_pred_subshots / num_pred_subshots
Subshot_F1        = F1(Subshot_Precision, Subshot_Recall)
Subshot           = Subshot_F1
```

`WHEN_S` 是匹配的 GT 子镜头时间段与其匹配的预测子镜头组时间并集之间的平均联合 IoU。

### 3.8 WHEN

紧凑时间指标为：

```text
WHEN = mean(WHEN_H, WHEN_E, WHEN_S)
```

仅适用的组件被纳入均值计算。

### 3.9 EVSA

`EVSA` 衡量事件-镜头关联是否被正确重建。它首先将预测事件和预测镜头映射到 GT ID 空间，然后比较事件-镜头边。

GT 边来自：

```text
gt_shot.active_events
```

预测边来自：

```text
pred_shot.active_events
```

仅当事件和镜头均成功匹配到 GT 时，预测边才能参与计算。

扩展指标为：

```text
EVSA_P = correct_mapped_pred_edges / mapped_pred_edges
EVSA_R = correct_mapped_pred_edges / gt_edges
EVSA   = F1(EVSA_P, EVSA_R)
```

主表 `EVSA` 使用 F1。

---

## 4. LLM-as-Judge 指标

`mtss_llm_judge.py` 是一个独立的评估器，不修改规则指标表，也不输出聚合分数。

它首先规范化 GT 和预测，然后复用 `mtss_metrics.py` 中的对齐工具：

```text
match_references
match_events
match_shots
```

由于 `match_references` 现已是 LLM 辅助匹配，LLM 评判在任务收集阶段也使用 LLM 辅助引用匹配。

### 4.1 引用质量

引用质量评判针对匹配的人物引用和匹配的场景引用。每对匹配配对被评分两次：

```text
LLM_person_ref_recall
LLM_person_ref_precision
LLM_scene_ref_recall
LLM_scene_ref_precision
```

**召回方向**：以 GT `detail_description` 为主体，检查预测的描述是否覆盖了 GT 中的所有事实内容。评分关注 GT 中的每个具体事实元素（如服装颜色、发型、体型、场景布局、光照条件）是否在预测中被提及。遗漏被扣分；矛盾被扣更多分。

**精确方向**：以预测的 `detail_description` 为主体，检查 GT 的描述是否覆盖了预测中的所有事实内容。预测中存在但 GT 中不存在的事实被视为潜在幻觉/无支撑声明。矛盾比无支撑声明更严重。

每个方向仅比较：

```text
appearance_anchor.id_features.detail_description
```

LLM 输出 1 到 5 的分数。聚合时，分数被归一化到 `[0, 1]`：

```text
normalized = (score - 1) / 4
```

紧凑字段为召回和精确的均值：

```text
LLM_person_ref = mean(LLM_person_ref_recall, LLM_person_ref_precision)
LLM_scene_ref  = mean(LLM_scene_ref_recall, LLM_scene_ref_precision)
```

物体引用参与引用映射并影响下游规则指标和部分上下文，但当前 LLM 评判表没有单独的 `LLM_object_ref` 字段。

### 4.2 事件质量

`LLM_event` 仅针对匹配的非对话事件。

它比较：

```text
content.description
```

评分关注 GT 描述中的所有事实元素是否被预测覆盖，包括：主要动作/活动、涉及的实体及其角色、实体间的空间关系、因果或顺序子动作、与事件相关的环境/音频上下文、以及结果或状态变化。每个事实元素是一个独立的覆盖点。

矛盾（陈述相反内容）比遗漏更严重。预测中未在 GT 提及的额外细节既不扣分也不加分。

不评分对话行、事件类型准确性、事件时间或镜头位置。引用映射作为上下文提供给 LLM，帮助其理解事件文本中的 ID。

### 4.3 子镜头 LLM 二分类匹配

LLM 子镜头匹配是一个局部二分类任务，与 1-5 分的子镜头语义质量评分分开。

两个二分类维度为：

```text
LLM_subshot_match_precision
LLM_subshot_match_recall
```

预测侧精确方向：每个预测子镜头作为目标。代码找到其所在的匹配 GT 镜头，根据子镜头索引投影出一个小的 GT 候选组，然后询问 LLM 该目标预测是否至少有一个真实对应项。

GT 侧召回方向：每个 GT 子镜头作为目标。代码找到其所在的匹配预测镜头，根据子镜头索引投影出一个小的预测候选组，然后询问 LLM 该目标 GT 是否至少有一个真实对应项。

候选组大小由两个半径参数控制：

```text
SHOT_CONTEXT_RADIUS = 2
SUBSHOT_CONTEXT_RADIUS = 2
```

`SHOT_CONTEXT_RADIUS` 确定从匹配镜头在对侧镜头序列中向前向后扩展多少个镜头以收集候选子镜头。`SUBSHOT_CONTEXT_RADIUS` 确定从候选中的投影位置向前向后取多少个子镜头作为最终候选组。

LLM 提示要求仅输出：

```text
<match>1</match>
```

或：

```text
<match>0</match>
```

此 LLM 子镜头匹配不是一对一匹配。它确定每个目标在局部候选组中是否至少有一个对应状态。聚合时，匹配的 GT 键和匹配的预测键被去重。

输出摘要保存：

```text
llm_subshot_local_matching.Subshot_Recall
llm_subshot_local_matching.Subshot_Precision
llm_subshot_local_matching.Subshot_F1
llm_subshot_local_matching.num_gt_subshots
llm_subshot_local_matching.num_pred_subshots
llm_subshot_local_matching.num_matched_gt_subshots
llm_subshot_local_matching.num_matched_pred_subshots
```

### 4.4 子镜头语义质量

1-5 分的子镜头语义质量评分以二分类匹配结果为门控。

仅当目标的 `match=1` 时才进一步计算：

```text
LLM_subshot_precision
LLM_subshot_recall
```

评估的事实元素包括：哪些实体可见（带 ref ID 的人物/物体）、当前姿势/动作/手势/表情、实体间的空间位置和关系、与物体或环境的交互、移动方向/速度、以及任何瞬态视觉状态（如光照变化、物体出现/消失）。每个事实元素是一个独立的覆盖/验证点。

精确方向：以预测子镜头为目标，检查其事实是否被 GT 上下文支撑。预测中未在 GT 提及的细节不扣分；仅与 GT 矛盾的才扣分。

召回方向：以 GT 子镜头为目标，检查预测上下文是否覆盖 GT 中的所有事实。遗漏和矛盾均被扣分，矛盾更严重。

紧凑字段：

```text
LLM_subshot
```

为精确侧和召回侧语义分数合并原始 1-5 分后统一归一化的均值：

```text
LLM_subshot = (mean(all_precision_scores + all_recall_scores) - 1) / 4
```

### 4.5 镜头事实质量

`LLM_shot_fact` 针对匹配的镜头配对，仅在存在可比较的镜头事实字段时创建任务。

它仅评分：

```text
camera
shot_transfer
```

评分关注 GT 中的所有镜头/转场事实是否被预测覆盖，包括：镜头景别（特写/中景/全景/远景）、拍摄角度（俯拍/仰拍/平视/鸟瞰/荷兰角）、镜头运动（摇/倾/推/拉/跟/摇臂/手持/固定）、构图（三分法/居中/偏移）、景深、焦点主体、转场类型（硬切/叠化/淡入淡出/划变/匹配剪辑）、以及转场速度。

矛盾比遗漏更严重。预测中的额外细节不扣分。

不评分镜头视觉描述、活跃事件、镜头内引用、事件内容或时间。

---

## 5. 批处理脚本行为

`scripts/eval_all_metrics.sh` 支持：

```text
all
rule-base
rule-based
llm-only
convert-only
```

脚本通过以下变量选择评判端点，全部在脚本内带默认值：

```text
JUDGE_BASE_URL      OpenAI 兼容端点   （默认 http://localhost:8000/v1）
JUDGE_MODEL         评判模型名        （默认 gpt-4o）
JUDGE_API_KEY       API key           （默认 EMPTY）
JUDGE_WORKERS       评判并发请求数    （默认 8）
JUDGE_TEMPERATURE   采样温度          （默认 0.7）
JUDGE_MAX_TOKENS    最大输出 token    （默认 4096）
METRICS_WORKERS     规则指标并发数    （默认 4）
```

任何 OpenAI 兼容服务均可作为评判端点：OpenAI、OpenRouter、DashScope，或自托管 vLLM。

选定的端点同时控制：

- `mtss_llm_judge.py` 中的 LLM 评判调用，通过命令行参数传入。
- `mtss_metrics.py` 中的 LLM 辅助引用匹配，通过 `REF_MATCH_*` 组传入。

仅运行 LLM 评判时，`mtss_llm_judge.py` 会在内部设置同一个 `REF_MATCH_*` 组，使任务收集阶段的引用匹配使用评判模型。

启动 LLM-only 指标时，脚本通过命令行参数将选定的后端和模型传递给 `mtss_llm_judge.py`；该客户端在内部为任务收集阶段的引用匹配设置相同的 `REF_MATCH_*` 组。

Transformer 字幕基线（如需要）首先通过 `mtss_transformer_convert.py` 转换为 MTSS 格式。转换输出写入：

```text
results_mtss_converted.jsonl
```

非 Transformer 模型直接使用：

```text
results.jsonl
```

---

## 6. 输出字段

规则指标主表为：

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

规则指标详情文件包含以下扩展字段：

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

LLM 评判表报告：

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

LLM 子镜头局部匹配块单独报告二分类匹配获得的召回率、精确率和 F1。这与 1-5 分的子镜头语义质量分数不同——它们代表两组不同的信息。

---

## 7. 当前未报告的指标

当前代码有意不报告：

- 聚合总分。
- 独立的事件类型准确率。
- 独立的镜头边界误差诊断。
- 预测侧幻觉或无支撑额外信息率。
- 对话 LLM 评判分数。
- `LLM_object_ref`。

这些选择旨在保持主表紧凑，并确保每个数字直接对应 MTSS 状态的一个明确定义的组件。
