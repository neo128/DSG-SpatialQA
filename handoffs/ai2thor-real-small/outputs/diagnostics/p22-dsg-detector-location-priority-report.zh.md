# P22 DSG detector current-location 优先级报告

## 背景

本阶段仍遵循“先提高 VLM-only，再优化 DSG”的顺序，但 P21 的真实 VLM retry prediction 尚未返回。因此本阶段不更新 VLM-only 成功率，也不声称 VLM-only 已提升。

当前 VLM-only 状态：

- P21 ready35 scoped retry expected：35；
- retry received：0；
- missing retry：35；
- out-of-scope failed case：1（缺 target crop 的 `cd`）；
- 当前 pending VLM-only semantic match：24/60 = 0.40。

## 本阶段问题

重新评估当前 GraphTool candidate 后发现，旧 P4 candidate 已经达到：

- semantic match：55/60；
- strict exact match：44/60。

剩余 5 个失败全部集中在 `FloorPlan201` 的 step `200007`，失败模式一致：图中有显式 detector current-location 边，例如 `book ON chair`、`creditcard ON chair`、`laptop ON chair`、`newspaper ON chair`、`diningtable ON chair`，但旧 candidate 被几何 inferred containment 或 support-like room fallback 盖过，回答成 `ON diningtable`、`ON floor` 或 `IN_ROOM`。

## 原理和修复

本阶段没有引入外部模型，也没有调用网络。修复分两层：

1. 先用当前代码和 `predicted-graph-p3-offset.json` 重新生成 GraphTool candidate，消除旧 candidate artifact 中 stale location 排序的影响。
2. 再修正 QA 层 support-like target 规则：对 `diningtable/chair/cabinet` 等支撑类目标，默认仍保留“优先回答 room”的保守策略；但如果同 step 存在显式 `source="detector_current_location"` 且 `source_kind="detector"` 的 location edge，则优先使用该显式 detector evidence。

这个修复避免把普通几何推断当成强证据，同时允许真实 detector/RGB-D current-location 覆盖保守 room fallback。

## 新增产物

- P22 candidate prediction：
  `handoffs/ai2thor-real-small/inputs/candidate/predicted-graph-tool-observation-aware-p22-detector-location-priority.jsonl`
- P22 semantic eval：
  `handoffs/ai2thor-real-small/outputs/diagnostics/dsg-semantic-eval-p22-detector-location-priority.json`
- P22 vs VLM pending semantic delta：
  `handoffs/ai2thor-real-small/outputs/diagnostics/dsg-p22-vs-vlm-p21-pending-semantic-delta.json`
- P22 QA eval：
  `handoffs/ai2thor-real-small/outputs/diagnostics/dsg-qa-eval-p22-detector-location-priority.json`
- P22 vs VLM pending QA delta：
  `handoffs/ai2thor-real-small/outputs/diagnostics/dsg-p22-vs-vlm-p21-pending-qa-delta.json`

## 结果

P22 DSG / GraphTool 结果：

- case count：60；
- semantic match：60/60 = 1.00；
- strict exact match：60/60 = 1.00；
- standard QA exact match：60/60 = 1.00；
- semantic eval digest：`e2f24bc5fbfee24d1af01b2cf42904b6b40a89f07ab1002fdd420a8cd154d425`。
- QA eval digest：`d7fb7b31e74e157cf835b38813c38e71dff118ea57f7cc433d896b1968ba03d0`。

相对当前 pending VLM-only：

- VLM-only semantic match：24/60 = 0.40；
- VLM-only standard QA exact match：0/60 = 0.00；
- DSG semantic match：60/60 = 1.00；
- DSG standard QA exact match：60/60 = 1.00；
- semantic delta：+36 cases，+0.60；
- QA exact delta：+60 cases，+1.00；
- paired wins/losses/ties：36 / 0 / 24；
- delta decision：`candidate_improved`；
- delta digest：`2fdf324aab007592d310c3d877aed72f988f409d9e22b76e209d00997a1f60b8`。

## 结论

在当前 observation-aware 60 QA slice 上，使用现有本地 observation-backed predicted graph 和 P22 GraphTool candidate，DSG 已经达到 60/60，并明显高于当前 pending VLM-only 的 24/60。

但这仍不是最终“强 VLM-only 后”的研究结论，因为 P21 ready35 的真实 VLM retry prediction 还没有返回。最终结论需要在收到 `vlm-p21-ready35.jsonl` 后，重新 merge/eval VLM-only，再与 P22 DSG 结果做正式比较。

当前阶段可确认：

- DSG 侧的 DS/GraphTool 已完成一轮有效优化；
- 当前 VLM-only 提升仍被外部 retry prediction 缺失阻塞；
- 下一步应优先接收或重新授权运行 35 条 P21 VLM retry，然后复算最终 delta。
