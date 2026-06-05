# p7 DSG Location Priority 阶段报告

## 目标

本阶段目标是在 p6 已解决 room/floor 语义兼容后，继续优化 DSG/DS 侧剩余的 5 个 `destination_mismatch`。这些错误集中在 `FloorPlan201`，表现为 gold 是 `ON chair`，GraphTool candidate 却回答 `ON diningtable` 或 `ON floor`。

## 根因

排查 predicted graph 后发现，图中同一步同时存在两类位置边：

1. 显式 detector current-location 边：
   - 例如 `book ON chair`
   - attributes 包含 `source="detector_current_location"`、`source_kind="detector"`

2. 几何推断 containment 边：
   - 例如 `book ON diningtable`
   - attributes 包含 `source="containment_inference"`、`inferred=true`

原 `SpatialQAEngine._latest_location_edge()` 和 `GraphTool._latest_location*()` 只按普通 edge sort 取最后一条。于是当同 step 出现多条 `ON` 边时，字典序靠后的 `diningtable`/`floor` 可能覆盖 detector 明确给出的 current location。

这不是 detector 没有正确边，而是 current-location selection 没有区分显式 detector evidence 和几何推断 evidence。

## 修改

新增 location-edge 专用排序：

1. 优先较新的 `step`；
2. 同 step 下优先 `source="detector_current_location"`；
3. 其次优先其他 `source_kind="detector"`；
4. 最后才使用普通 inferred/geometry edge。

修改范围：

- `src/dsg_spatialqa_lab/qa.py`
- `src/dsg_spatialqa_lab/graph_tool.py`
- `tests/test_spatial_qa.py`

新增测试覆盖：

- QA object_location 在同 step 冲突边中优先 detector current location；
- GraphTool object_timeline 同样优先 detector current location；
- 没有 detector current location 时，仍然可以使用 inferred location。

## p7 指标

使用同一 `observation-aware-p4-target60` 60 条 object-location QA，重跑 GraphTool candidate：

| 系统 | Semantic Match | Rate |
| --- | ---: | ---: |
| DSG GraphTool p7 location priority | 60/60 | 1.000000 |
| DSG GraphTool p6 room/floor compatible | 55/60 | 0.916667 |
| VLM-only qwen3.7 p4 | 17/60 | 0.283333 |
| Multi-frame VLM qwen3.7 window4 p4 | 25/60 | 0.416667 |

Delta：

| 对比 | Wins | Losses | Ties | Rate Delta |
| --- | ---: | ---: | ---: | ---: |
| DSG p7 vs VLM-only | 43 | 0 | 17 | +0.716667 |
| DSG p7 vs Multi-frame VLM | 35 | 0 | 25 | +0.583333 |

## 修复的 5 个案例

本次修复全部 5 个 p6 剩余错误：

- `book`: `ON diningtable` -> `ON chair`
- `creditcard`: `ON diningtable` -> `ON chair`
- `diningtable`: `ON floor` -> `ON chair`
- `laptop`: `ON diningtable` -> `ON chair`
- `newspaper`: `ON diningtable` -> `ON chair`

## Artifact

- p7 candidate:
  `handoffs/ai2thor-real-small/inputs/candidate/predicted-graph-tool-observation-aware-p7-location-priority.jsonl`
- semantic eval:
  `handoffs/ai2thor-real-small/outputs/diagnostics/dsg-candidate-semantic-eval-p7-location-priority.json`
- delta vs VLM-only:
  `handoffs/ai2thor-real-small/outputs/diagnostics/dsg-candidate-semantic-delta-vs-vlm-p7-location-priority.json`
- delta vs multi-frame VLM:
  `handoffs/ai2thor-real-small/outputs/diagnostics/dsg-candidate-semantic-delta-vs-multi-frame-vlm-p7-location-priority.json`

## 结论

p7 证明：DSG/DS 剩余错误的主要根因不是图中缺失正确支撑物，而是同 step 多条位置边的选择策略没有优先 detector current-location evidence。修复后，当前 observation-aware 诊断切片上 DSG GraphTool 达到 `60/60`。

但这仍不是最终 real research-ready 结论。原因是：

1. 当前 p7 使用的是 observation-aware QA 诊断切片；
2. VLM-only 还需要用 p3 answer-options bundle 真实重跑；
3. 最终结论必须重新通过 readiness gate，并基于真实 external VLM/LLM predictions、detector/RGB-D evidence、QA eval/delta、graph eval 和 error attribution。

下一步应在明确授权外部 VLM/API 后，用 p3 bundle 重跑 VLM-only 和 multi-frame VLM，然后重新计算 DSG p7 vs stronger VLM baseline 的 delta。
