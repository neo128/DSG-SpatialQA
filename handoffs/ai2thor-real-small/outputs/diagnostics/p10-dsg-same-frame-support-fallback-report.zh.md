# p10 DSG 同帧唯一支撑物 fallback 优化报告

## 目标

p9 诊断显示，p8 independent detector-backed DSG 的剩余 `ON <support>` 失败可拆成：

- `target_missing=15`
- `support_missing=14`
- `support_present_but_relation_missing=6`

p10 只处理第三类中最安全的一部分：目标和支撑物都已经由 detector/RGB-D 证据进入 predicted graph，且在同一 scene、同一 step 中只有一个可信 `ON` 支撑候选。

## 方法

`SpatialQAEngine._answer_object_location()` 的 fallback 顺序现在是：

1. 优先使用显式 containment edge；
2. 如果没有显式位置边，尝试同帧唯一 detector support fallback；
3. 如果仍无法确定，再退回 p8 的 room-level fallback。

同帧 support fallback 的触发条件：

- 目标节点来自 detector；
- 目标和支撑物都具备 `rgb/depth/detector` evidence；
- 二者 `visible=true`；
- 二者同 `scene_id`、同 `step`；
- 支撑物 label 属于 ON surface 类；
- 目标本身不是家具/支撑物/容器类；
- 候选支撑物唯一；
- XZ 平面距离在支撑物尺度阈值内。

为了防止误判，本阶段明确禁止：

- 把 `cabinet/drawer/fridge/sink` 等容器当成 `ON` 支撑；
- 把 `chair/table/bed` 等支撑物目标再推断成 `ON` 到其它支撑物；
- 多支撑候选时猜最近。

## 产物

- prediction：`handoffs/ai2thor-real-small/inputs/candidate/predicted-graph-tool-independent-p10-same-frame-support.jsonl`
- semantic eval：`handoffs/ai2thor-real-small/outputs/diagnostics/dsg-candidate-semantic-eval-p10-same-frame-support-independent.json`
- support gap：`handoffs/ai2thor-real-small/outputs/diagnostics/dsg-support-gap-p10-same-frame-support-independent.json`
- delta vs VLM：`handoffs/ai2thor-real-small/outputs/diagnostics/dsg-candidate-semantic-delta-vs-vlm-p10-same-frame-support-independent.json`
- delta vs multi-frame VLM：`handoffs/ai2thor-real-small/outputs/diagnostics/dsg-candidate-semantic-delta-vs-multi-frame-vlm-p10-same-frame-support-independent.json`
- delta vs p8：`handoffs/ai2thor-real-small/outputs/diagnostics/dsg-candidate-semantic-delta-p10-vs-p8-independent.json`

## 结果

| 指标 | p8 room fallback | p10 same-frame support |
| --- | ---: | ---: |
| Semantic match | 20/60 | 23/60 |
| Semantic match rate | 0.333333 | 0.383333 |
| Delta | - | +3 |

相对当前 independent VLM baselines：

| Baseline | Baseline semantic | p10 DSG semantic | Delta |
| --- | ---: | ---: | ---: |
| VLM p4 independent | 6/60 | 23/60 | +17 |
| Multi-frame VLM p4 independent | 4/60 | 23/60 | +19 |

p10 support gap：

| 缺口类型 | p9 | p10 |
| --- | ---: | ---: |
| `target_missing` | 15 | 15 |
| `support_missing` | 14 | 14 |
| `support_present_but_relation_missing` | 6 | 3 |

## 解释

p10 只救回 3 条，是刻意保守的结果。它没有尝试用 gold support label 猜答案，也没有在多支撑候选或跨 step 目标/支撑物之间硬连边。

剩余 32 条 `ON` 失败中，29 条仍属于 detector coverage 问题：

- 15 条目标没进 predicted graph；
- 14 条支撑物没进 predicted graph。

因此下一步收益最高的不是继续扩大 fallback，而是补 detector/RGB-D coverage，尤其是：

- `countertop`
- `shelf`
- `coffeetable`
- `dresser`
- `handtowelholder`

## 阶段结论

p10 是一个小但可靠的 DS 优化：

- independent detector DSG 从 20/60 提升到 23/60；
- support-present relation gap 从 6 降到 3；
- 没有把 mock/gold 注入 candidate；
- 没有调用外部模型或网络。

但这还不是最终研究结论。VLM-only 仍需要使用 p5 image-role prompt 与 p4 option-id contract 做真实重跑；DSG 还需要补 detector coverage 后再跑 p11/p12。
