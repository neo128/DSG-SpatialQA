# P22 VLM-only room answer-option fallback 报告

## 背景

本阶段目标是先提高 VLM-only 成功率，再继续优化 DSG。P21 ready35 的真实
VLM retry prediction 仍未返回，因此本阶段不调用外部 VLM/LLM，也不把 pending
retry 伪装成真实返回。

本轮只做一个 no-gold、确定性的后处理：如果某个 VLM prediction 没有可用
`current_location`，并且对应 request bundle 中只有一个公开的 `answer_options`
候选，且该候选严格为 `IN_ROOM / room / fallback_room`，则把该候选写回为
结构化 answer。多候选、非 room fallback、已有 current_location 的 case 全部跳过。

## 原理

这个 fallback 不读取 gold answer，也不读取 gold evidence。它只使用已经发给外部
VLM 的公开 request bundle 字段：

- `case_id`
- `primary_frame.step`
- `answer_options[0].option_id`
- `answer_options[0].relation`
- `answer_options[0].destination_label`
- `answer_options[0].source`

应用条件：

- prediction 没有可解析的 `current_location`；
- request bundle 中该 case 恰好只有一个 answer option；
- option 必须是 `relation="IN_ROOM"`、`destination_label="room"`、
  `source="fallback_room"`；
- fallback 成功后清除原 prediction 的 `error`，因为后处理已经给出可评估结构化 answer；
- 原始 VLM 的文本说明和 evidence 字段保留。

## 新增产物

| artifact | path |
| --- | --- |
| P22 fallback VLM prediction | `inputs/offline-controls/reruns/observation-aware-p4-target60/vlm-p22-room-option-fallback.jsonl` |
| fallback report | `outputs/diagnostics/vlm-p22-room-option-fallback-report.json` |
| P22 VLM semantic eval | `outputs/diagnostics/vlm-semantic-eval-p22-room-option-fallback.json` |
| P22 VLM QA eval | `outputs/diagnostics/vlm-qa-eval-p22-room-option-fallback.json` |
| P22 VLM vs P21 pending semantic delta | `outputs/diagnostics/vlm-p22-room-option-fallback-vs-p21-pending-semantic-delta.json` |
| DSG P22 vs VLM P22 semantic delta | `outputs/diagnostics/dsg-p22-vs-vlm-p22-room-option-fallback-semantic-delta.json` |

## 数据

fallback report：

| metric | value |
| --- | ---: |
| input predictions | 60 |
| case inputs in P21 ready35 bundle | 35 |
| applied fallback | 5 |
| skipped existing location | 11 |
| skipped no single room option | 19 |
| missing case input | 25 |

applied case ids：

- `ai2thor-real-small-episode-002:FloorPlan201:0002:object_location:armchair_04_38_00_00_06_02:observation_aware:200013`
- `ai2thor-real-small-episode-003:FloorPlan301:0002:object_location:baseballbat_02_84_00_29_01_82:observation_aware:200012`
- `ai2thor-real-small-episode-004:FloorPlan401:0009:object_location:garbagecan_00_05_00_00_03_88:observation_aware:200014`
- `ai2thor-real-small-episode-005:FloorPlan2:0008:object_location:cabinet_00_63_02_02_01_45:observation_aware:100047`
- `ai2thor-real-small-episode-005:FloorPlan2:0010:object_location:cabinet_01_70_02_02_01_45:observation_aware:100047`

VLM-only semantic：

| version | semantic match | semantic rate | strict exact |
| --- | ---: | ---: | ---: |
| P21 scoped pending merge | 24 / 60 | 0.400000 | 0 / 60 |
| P22 room option fallback | 29 / 60 | 0.483333 | 0 / 60 |

P22 VLM vs P21 pending：

| metric | value |
| --- | ---: |
| semantic count delta | +5 |
| semantic rate delta | +0.083333 |
| paired wins / losses / ties | 5 / 0 / 55 |
| decision | `candidate_improved` |

DSG P22 vs stronger VLM P22：

| method | semantic match | semantic rate |
| --- | ---: | ---: |
| VLM-only P22 room fallback | 29 / 60 | 0.483333 |
| coverage-merged DSG / GraphTool P22 | 60 / 60 | 1.000000 |

| metric | value |
| --- | ---: |
| semantic count delta | +31 |
| semantic rate delta | +0.516667 |
| paired wins / losses / ties | 31 / 0 / 29 |
| decision | `candidate_improved` |

## 结论

本轮把当前可审计的 VLM-only semantic match 从 24/60 提高到 29/60，提升来自
5 条 no-gold single-room answer option fallback。这个提升是确定性的、可复验的，
没有调用外部模型，也没有使用 gold answer。

在这个更强的 VLM-only P22 baseline 下，coverage-merged DSG / GraphTool P22 仍为
60/60，semantic delta 为 +31 case。阶段性结论是：当前 observation-aware 60 QA
slice 上，DSG 仍明显优于已做 no-gold 后处理的 VLM-only。

限制：

- P21 ready35 的真实 VLM retry prediction 仍未返回；
- 标准 QA exact 对 VLM-only 仍是 0/60，因为 VLM answer 结构与 gold JSON 不做完全等价；
- 因此这仍是阶段性对比，不是最终强 VLM retry 后的研究结论。

关键 digest：

- fallback report digest：`3b7d8c12ae0c5a2b366fb9ffcecea13315ff68503b0c54554524167313528445`
- P22 VLM semantic eval digest：`9d9010f84d5e8ff8d4d9539d21e853d8d6776391ef3bb8d45004033d2dbbc138`
- P22 VLM delta digest：`8fca497a56e1c93df32dd0aa52be8c48f892649bb3c5b61d6e8947763fea206f`
- DSG vs VLM P22 delta digest：`b5ac0f3237f095d68a779d89e780403a71b4d173a1b8dc2ac17a0b0373ec5685`
