# DSG p8 detector room fallback 优化报告

## 目标

本阶段优化 DS/DSG 的独立 detector-backed 诊断切片。目标不是用 oracle metadata 补全支撑面，也不是猜 `ON` 关系，而是在真实 detector/RGB-D 证据已经确认目标可见、但 graph 缺 containment edge 时，给出一个保守的 room-level 位置 fallback。

## 根因

`dsg-candidate-semantic-eval-p4-target60-independent.json` 显示 independent detector DSG 的语义正确率为：

- 0/60。
- 主要失败为 `location_not_parseable`：39 条。

检查 predicted graph 后发现，许多目标对象已经作为 detector-backed node 进入图中，节点包含：

- `source_kind="detector"`
- `evidence_kinds=["depth","detector","rgb"]`
- `scene_id`
- `pose`
- `visible=true`

但这些对象没有 `IN_ROOM`、`ON`、`INSIDE` 等 containment edge。因此 GraphTool 能找到对象状态，却无法生成可解析的 `current_location`。

## 本阶段改动

`SpatialQAEngine._answer_object_location()` 现在在没有明确 location edge 时，会启用一个保守 fallback：

- 仅当对象节点来自 detector；
- 且 `visible=true`；
- 且包含 `scene_id`；
- 且 evidence kinds 同时包含 `rgb/depth/detector`；
- 且当前没有更具体的 containment edge。

满足以上条件时，返回：

```json
{"relation": "IN_ROOM", "dst": "ai2thor_room", "step": <state_step>}
```

这个 fallback 只表达“目标在当前观测场景/房间中被检测到”，不推断目标在桌上、床上、柜内等更具体支撑关系。

## 结果

重跑 independent p4 predicted graph candidate 后生成：

- prediction：`handoffs/ai2thor-real-small/inputs/candidate/predicted-graph-tool-independent-p8-room-fallback.jsonl`
- semantic eval：`handoffs/ai2thor-real-small/outputs/diagnostics/dsg-candidate-semantic-eval-p8-room-fallback-independent.json`
- delta vs VLM：`handoffs/ai2thor-real-small/outputs/diagnostics/dsg-candidate-semantic-delta-vs-vlm-p8-room-fallback-independent.json`
- delta vs multi-frame VLM：`handoffs/ai2thor-real-small/outputs/diagnostics/dsg-candidate-semantic-delta-vs-multi-frame-vlm-p8-room-fallback-independent.json`

核心指标：

| 指标 | p4 independent DSG | p8 room fallback DSG |
| --- | ---: | ---: |
| Semantic match | 0/60 | 20/60 |
| Semantic match rate | 0.000000 | 0.333333 |
| location_not_parseable | 39 | 0 |

与独立 VLM baseline 对比：

| Baseline | Baseline semantic | p8 DSG semantic | Delta |
| --- | ---: | ---: | ---: |
| VLM p4 independent | 6/60 | 20/60 | +14 |
| Multi-frame VLM p4 independent | 4/60 | 20/60 | +16 |

## 解释

p8 证明了一个重要点：独立 detector DSG 原先完全失败，不是因为所有对象都不可见，而是因为对象节点已经存在但缺少可解析 location。保守 room fallback 能把 room-level QA 先救回来，并让错误从“不可解析”变成更具体的“缺支撑面关系”。

剩余失败主要包括：

- `relation_mismatch`：20 条。多数是 gold 为 `ON <support>`，p8 只能给 `IN_ROOM room`。
- `Object not found`：约 19 条。说明仍需要提高 detector/object coverage。
- `destination_mismatch`：1 条。

## 阶段结论

这是一个真实有用但保守的 DS 优化：

- 它让 independent detector-backed DSG 从 0/60 提到 20/60。
- 它超过当前 independent VLM 和 multi-frame VLM baseline。
- 它不构成最终“DSG 已证明优于强 VLM”的结论，因为 VLM-only 还需要用 p5/p6 prompt 真实重跑，DSG 还需要补真实支撑面/containment edge。

下一步 DS 优化应该集中在：

- 用 detector/RGB-D 关系证据生成更可靠的 `ON`/`INSIDE` edge；
- 对 `Object not found` case 做 coverage collection；
- 在真实 VLM p5/p6 重跑后，与 p8/p9 DSG 在同一 QA slice 上重新比较。
