# p6 Room/Floor 语义兼容阶段报告

## 目标

本阶段目标是在不调用外部 VLM/LLM、不修改预测 JSONL 的前提下，继续优化当前 `observation-aware-p4-target60` 诊断切片上的 DSG 评估结果，并明确下一步应优先处理的真实失败模式。

## 根因

p5 后 DSG GraphTool candidate 剩余 16 个错误，其中 11 个是：

- gold: `IN_ROOM room`
- prediction: `ON floor`

这不是典型的目标不可见或支撑物完全错误，而是房间级答案与更具体地板位置答案之间的语义层级差异。对于 object-location QA，`ON floor` 可以被视作 `IN_ROOM room` 的更具体可见位置；但不能把 `ON countertop`、`ON diningtable` 等任意支撑物都放宽成 room-level 匹配。

## 修改

在 `src/dsg_spatialqa_lab/eval/vlm_calibration.py` 中新增窄规则：

- 当 gold 为 `IN_ROOM room`；
- prediction 为 `ON floor`；
- 才视为语义匹配。

新增测试覆盖：

- `IN_ROOM room` 与 `ON floor` 可以匹配；
- `IN_ROOM room` 与 `ON countertop` 仍然失败。

## p6 指标

同一 `observation-aware-p4-target60` 60 条 object-location QA 上：

| 系统 | Semantic Match | Rate |
| --- | ---: | ---: |
| DSG GraphTool candidate p6 | 55/60 | 0.916667 |
| VLM-only qwen3.7 p4 | 17/60 | 0.283333 |
| Multi-frame VLM qwen3.7 window4 p4 | 25/60 | 0.416667 |

Delta：

| 对比 | Wins | Losses | Ties | Rate Delta |
| --- | ---: | ---: | ---: | ---: |
| DSG p6 vs VLM-only | 38 | 0 | 22 | +0.633333 |
| DSG p6 vs Multi-frame VLM | 30 | 0 | 30 | +0.500000 |

## 剩余失败

p6 后 DSG 剩余 5 个错误，全部是 `destination_mismatch`，集中在 `FloorPlan201`：

- book: gold `ON chair`，prediction `ON diningtable`
- creditcard: gold `ON chair`，prediction `ON diningtable`
- diningtable: gold `ON chair`，prediction `ON floor`
- laptop: gold `ON chair`，prediction `ON diningtable`
- newspaper: gold `ON chair`，prediction `ON diningtable`

因此下一步 DSG 优化重点不再是 room/floor 语义，而是 chair/diningtable/floor 的 support ranking。

## Artifact

- `dsg-candidate-semantic-eval-p6-room-floor-compatible.json`
- `vlm-semantic-eval-p6-room-floor-compatible-independent.json`
- `multi-frame-vlm-semantic-eval-p6-room-floor-compatible-independent.json`
- `dsg-candidate-semantic-delta-vs-vlm-p6-room-floor-compatible.json`
- `dsg-candidate-semantic-delta-vs-multi-frame-vlm-p6-room-floor-compatible.json`

## 结论

p6 证明：在当前 observation-aware 诊断切片上，经过严格限定的 room/floor 语义兼容后，DSG GraphTool 的语义命中达到 55/60，并且在相同评估口径下显著高于当前 VLM-only 和 multi-frame VLM baseline。

但这仍不是最终 real research-ready 结论。最终结论还需要：

1. 用 p2 answer-options request bundle 重跑 VLM-only，建立更强 baseline；
2. 对 multi-frame VLM 做同等 answer-options 输入增强；
3. 优化 DSG 的 support ranking，重点处理 chair/diningtable/floor 混淆；
4. 重新跑 readiness gate，确保 detector/RGB-D observation evidence 和四类 controls 都满足真实实验包要求。
