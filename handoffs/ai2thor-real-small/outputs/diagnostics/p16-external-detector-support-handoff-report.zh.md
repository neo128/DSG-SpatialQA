# P16 外部 Detector 支撑关系 Handoff 报告

## 目标

P15 已证明：DSG 只要拿到足够的支撑物节点和 `ON` 边，在同一批 60 条
observation-aware QA 上可以从 23/60 提升到 43/60 或 59/60。本轮目标是把这个
诊断上界转成下一轮真实外部 detector/RGB-D 可执行的 handoff，而不是继续只停留在
AI2-THOR metadata coverage。

本轮没有调用外部 API、没有启动 detector、没有启动仿真器。

## 新增产物

1. 最新 support gap：
   - `dsg-support-gap-p16-independent-p4-target60.json`
2. schema-fixed frame index：
   - `vlm-frame-index-coverage-p16-schema-fixed-p2.jsonl`
   - `vlm-frame-index-coverage-p16-schema-fixed-p2-p3.jsonl`
3. VLM p7 支撑候选 request bundle：
   - `offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p7-schema-fixed-support-candidates.json`
4. VLM p7 增强 request bundle：
   - `offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p7-independent-support-crop-options.json`
5. 外部 detector 支撑物 handoff：
   - `independent-detector-rgbd-handoff-p16-schema-fixed-support.json`
6. P16 汇总 JSON：
   - `p16-external-detector-support-handoff-report.json`

## Support Gap

基于最新 independent DSG p14 失败集：

| 指标 | 数值 |
| --- | ---: |
| failed cases | 32 |
| support missing | 14 |
| support present but relation missing | 3 |
| target missing | 15 |

外部 detector 下一轮至少要补的支撑物标签：

- `coffeetable`
- `countertop`
- `dresser`
- `handtowelholder`
- `shelf`

## VLM-only 输入增强

P7 request bundle 在不暴露 gold answer / gold evidence 的前提下，补了更强的视觉输入契约：

| 指标 | 数值 |
| --- | ---: |
| QA cases | 60 |
| cases with answer options | 60 |
| cases with support candidates | 51 |
| cases with target crop | 41 |
| answer options | 184 |

说明：

- 支撑候选来自 schema-fixed frame index。
- target crop 仍使用 independent p4 detector bbox records。
- metadata-backed coverage records 没有 `bbox_2d_xyxy`，不能直接用于 target crop，这一点保留为真实 detector 的交付要求。

## Detector Handoff

新的外部 detector handoff 要求：

| 指标 | 数值 |
| --- | ---: |
| required frames | 28 |
| frames with support labels | 22 |
| support labels requested | 51 |

外部 detector/RGB-D 生产方下一轮需要在这些 frame 中输出：

- 目标物检测；
- support label 检测；
- 2D bbox；
- 3D bbox / pose；
- `rgb/depth/mask` evidence；
- `current_location_id/current_location_relation`，尤其是 `ON` 关系。

## 阶段结论

P16 把 P15 的“DSG 需要支撑关系才能赢”转成了可执行的外部 detector handoff。
现在下一步不再是继续改 GraphTool，而是让真实 detector/RGB-D 产出同等质量的支撑物和
`ON` 关系证据。

当前仍不能声称最终真实研究结论，因为 P15 的高分来自 metadata-backed coverage。
但 P16 已经明确了真实 detector 需要补什么，以及如何把补齐后的 JSONL 导回
SceneObservation sequence 和 predicted DSG。

## 下一步验收

1. 外部 detector 返回 `independent-detector-rgbd-handoff-p16-schema-fixed-support.json`
   要求的 detector JSONL。
2. 重新 import detector observations。
3. 重新 build observation-sequence-backed predicted graph。
4. 重新跑 GraphTool candidate。
5. 重新跑 VLM-only / multi-frame VLM p7 request bundle。
6. 只有在 external detector-backed DSG 仍然超过 VLM / multi-frame VLM 时，才写真实研究结论。
