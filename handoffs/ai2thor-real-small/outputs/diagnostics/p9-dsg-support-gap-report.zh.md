# p9 DSG 支撑关系缺口诊断报告

## 目标

本阶段继续优化 DS/DSG 的 independent detector-backed 诊断切片。p8 已经用保守 room fallback 把 `location_not_parseable` 修掉，但剩余错误集中在 `ON <support>` 支撑关系。p9 的目标是把这些失败拆成可行动的采集/建图缺口：

- 目标对象没有进入 predicted graph；
- 目标存在，但支撑物没有进入 predicted graph；
- 目标和支撑物都存在，但缺少 `ON` containment edge。

本报告是 evaluator-side 诊断，包含 gold support label，只能用于定位 DSG 缺口和规划 detector/RGB-D 补采集，不得放入外部 VLM request bundle。

## 产物

- JSON report：`handoffs/ai2thor-real-small/outputs/diagnostics/dsg-support-gap-p9-independent.json`
- schema：`dsg-spatialqa-lab.vlm-support-gap-report.v1`
- report digest：`25e08b0ace0cf3d6aedec97f3dd3e080a200638fc90b89b69224f9d226174bf2`

## 核心结果

在 p8 independent DSG 的失败样本中，`ON` 支撑关系相关失败共 35 条：

| 缺口类型 | 数量 | 含义 |
| --- | ---: | --- |
| `target_missing` | 15 | 目标对象未进入 predicted graph，必须先补 detector/object coverage |
| `support_missing` | 14 | 目标存在，但支撑物类别没有进入 predicted graph |
| `support_present_but_relation_missing` | 6 | 目标和支撑物都存在，但未生成 `ON` 边 |

需要优先补采集的支撑物类别：

- `countertop`
- `shelf`
- `coffeetable`
- `dresser`
- `handtowelholder`

支撑物已存在但缺关系边的类别：

- `chair`
- `bed`
- `bathtub`

## 解释

这说明 p9 之后的 DSG 优化不能只靠 QA fallback：

- `target_missing` 的 15 条必须靠更好的目标 detector coverage 解决。
- `support_missing` 的 14 条需要在 detector/RGB-D handoff 中显式要求检测支撑物，尤其是 `countertop`。
- `support_present_but_relation_missing` 的 6 条可以优先改建图逻辑：当目标和唯一可信支撑物同场景、同帧或稳定 track 且几何位置合理时，生成 `ON` edge。

## 下一步执行建议

1. 用 `dsg-support-gap-p9-independent.json` 生成 detector/RGB-D 补采集任务，优先补 `countertop`、`shelf`、`coffeetable`、`dresser`、`handtowelholder`。
2. 对 `chair`、`bed`、`bathtub` 三类 support-present case，加一个保守 `ON` edge 推断测试：只在唯一支撑物、同 scene、detector RGB-D evidence 完整、几何高度/平面重叠合理时生成。
3. 重新 build predicted graph，跑 p10 candidate semantic eval，目标是把 `support_present_but_relation_missing` 从 6 条降到 0 或接近 0。
4. VLM-only 侧需要用 p5 image-role + p4 option-id contract 真实重跑；当前 AGENTS 约束禁止外部 AI/API 调用，因此本阶段没有发起 VLM 网络请求。

## 阶段结论

p9 没有声称 DSG 最终优于 VLM。它把 DSG 剩余失败转成了可执行的工程任务：

- 先补 detector 覆盖解决 29/35 条；
- 再补几何 `ON` edge 解决 6/35 条；
- 等真实 VLM p5/p6 重跑后，再用同一 QA slice 做最终 DSG-vs-VLM 结论。
