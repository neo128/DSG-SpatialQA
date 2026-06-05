# P23/P24 VLM-only room-target 与 single-support fallback 报告

## 背景

本阶段继续执行“先提高 VLM-only，再优化 DSG”的顺序。P21 ready35 的真实 VLM
retry prediction 仍未返回，因此本轮不调用外部 VLM/LLM，也不把 pending retry
伪装成真实返回。

本轮在 P22 single-room option fallback 之后，又增加了两类 no-gold、确定性的
VLM-only 后处理：

1. P23 room-level target fallback：
   - 当目标是房间级/大件目标，例如 `cabinet`、`armchair`、`floor`；
   - VLM prediction 为 `target_not_observed` 或 `relation_not_observed`；
   - prediction 没有可用 `current_location`；
   - request bundle 中存在公开的 `IN_ROOM / room / fallback_room` option；
   - 则补为 `IN_ROOM room`。
2. P24 single-support option fallback：
   - 当目标不是房间级目标；
   - VLM prediction 为 `target_not_observed` 或 `relation_not_observed`；
   - prediction 没有可用 `current_location`；
   - request bundle 中恰好只有一个非 room `support_candidate`，并且有一个
     `IN_ROOM / room / fallback_room`；
   - 则补为该唯一 support option。

这两个后处理都只使用外部 VLM request bundle 中已经公开的 answer options，不读取
gold answer 或 gold evidence。

## 新增产物

| artifact | path |
| --- | --- |
| P23 prediction | `inputs/offline-controls/reruns/observation-aware-p4-target60/vlm-p23-room-level-target-fallback.jsonl` |
| P23 fallback report | `outputs/diagnostics/vlm-p23-room-level-target-fallback-report.json` |
| P23 semantic eval | `outputs/diagnostics/vlm-semantic-eval-p23-room-level-target-fallback.json` |
| P23 QA eval | `outputs/diagnostics/vlm-qa-eval-p23-room-level-target-fallback.json` |
| P24 prediction | `inputs/offline-controls/reruns/observation-aware-p4-target60/vlm-p24-single-support-option-fallback.jsonl` |
| P24 fallback report | `outputs/diagnostics/vlm-p24-single-support-option-fallback-report.json` |
| P24 semantic eval | `outputs/diagnostics/vlm-semantic-eval-p24-single-support-option-fallback.json` |
| P24 QA eval | `outputs/diagnostics/vlm-qa-eval-p24-single-support-option-fallback.json` |
| P24 vs P21 semantic delta | `outputs/diagnostics/vlm-p24-single-support-option-fallback-vs-p21-pending-semantic-delta.json` |
| DSG P22 vs VLM P24 semantic delta | `outputs/diagnostics/dsg-p22-vs-vlm-p24-single-support-option-fallback-semantic-delta.json` |

## P23 数据

P23 room-level target fallback applied cases：

- `ai2thor-real-small-episode-001:FloorPlan1:0008:object_location:cabinet_00_68_02_02_02_46:observation_aware:100040`
- `ai2thor-real-small-episode-001:FloorPlan1:0009:object_location:cabinet_00_72_02_02_02_46:observation_aware:100039`
- `ai2thor-real-small-episode-001:FloorPlan1:0010:object_location:cabinet_00_73_02_02_02_46:observation_aware:100040`
- `ai2thor-real-small-episode-002:FloorPlan201:0001:object_location:armchair_00_85_00_00_05_98:observation_aware:200001`
- `ai2thor-real-small-episode-004:FloorPlan401:0008:object_location:floor_01_44_00_04_01_85:observation_aware:100045`

P23 结果：

| version | semantic match | semantic rate | strict exact |
| --- | ---: | ---: | ---: |
| P22 VLM room option fallback | 29 / 60 | 0.483333 | 0 / 60 |
| P23 room-level target fallback | 34 / 60 | 0.566667 | 0 / 60 |

P23 vs P22：

| metric | value |
| --- | ---: |
| semantic count delta | +5 |
| semantic rate delta | +0.083333 |
| paired wins / losses / ties | 5 / 0 / 55 |
| decision | `candidate_improved` |

## P24 数据

P24 single-support fallback applied cases：

- `ai2thor-real-small-episode-004:FloorPlan401:0005:object_location:dishsponge_03_20_00_86_00_02:observation_aware:100044`
- `ai2thor-real-small-episode-004:FloorPlan401:supplemental_object_location:0016:papertowelroll_03_15_00_96_00_28:observation_aware:100044`

P24 结果：

| version | semantic match | semantic rate | strict exact |
| --- | ---: | ---: | ---: |
| P21 scoped pending merge | 24 / 60 | 0.400000 | 0 / 60 |
| P22 room option fallback | 29 / 60 | 0.483333 | 0 / 60 |
| P23 room-level target fallback | 34 / 60 | 0.566667 | 0 / 60 |
| P24 single-support fallback | 36 / 60 | 0.600000 | 0 / 60 |

P24 vs P21 pending：

| metric | value |
| --- | ---: |
| semantic count delta | +12 |
| semantic rate delta | +0.200000 |
| paired wins / losses / ties | 12 / 0 / 48 |
| decision | `candidate_improved` |

DSG P22 vs stronger VLM P24：

| method | semantic match | semantic rate |
| --- | ---: | ---: |
| VLM-only P24 | 36 / 60 | 0.600000 |
| coverage-merged DSG / GraphTool P22 | 60 / 60 | 1.000000 |

| metric | value |
| --- | ---: |
| semantic count delta | +24 |
| semantic rate delta | +0.400000 |
| paired wins / losses / ties | 24 / 0 / 36 |
| decision | `candidate_improved` |

## 结论

本轮把当前可审计 VLM-only semantic match 从 P22 的 29/60 继续提高到 P24 的
36/60；相对 P21 pending 的 24/60，累计提升 12 条。所有新增提升都来自公开
answer options 和 prediction 字段，不使用 gold answer。

在这个更强的 VLM-only P24 baseline 下，coverage-merged DSG / GraphTool P22 仍为
60/60，semantic delta 为 +24 条。阶段性结论是：当前 observation-aware 60 QA
slice 上，DSG 仍明显优于已做三轮 no-gold 后处理的 VLM-only。

限制：

- P21 ready35 的真实 VLM retry prediction 仍未返回；
- P24 使用 answer-option constrained 后处理，已经不是“裸 VLM-only”；
- 标准 QA exact 对 VLM-only 仍是 0/60，因为 VLM answer 结构与 gold JSON 不做完全等价；
- 因此最终结论仍需要真实 VLM retry 返回后复算。

关键 digest：

- P23 fallback report digest：`2589bbaaa145f6248e4b47f2827be9c6723295959b3c5976bb86c558b9e73daf`
- P23 semantic eval digest：`a7b6242edbcddb7d0b28990fc8d8ed314f1582a7373376d8470210549d50c3e9`
- P24 fallback report digest：`cfb363483fb6d2c5a2ccf9f00abdc68cc83278f4a54568814528a05e64ea57e3`
- P24 semantic eval digest：`9f998f31fbbc066c30c273c572c26f1e1956b678e6ee3ebf197ab73d1b933745`
- P24 vs P21 delta digest：`909da52fc6b207a3c853c6cda9770073715a7cd27efa266fc9a11d649d146565`
- DSG P22 vs VLM P24 delta digest：`a3051b1bea0ba16a1fd442264dd20f3fb217caaf49f0530fc0872c39768e061e`
