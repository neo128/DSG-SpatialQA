# VLM-only p5/p6 图片角色与选项策略优化报告

## 目标

本阶段目标是提高下一轮真实 VLM-only 重跑的输入质量。重点不是修改评估口径，也不是把旧 VLM 输出“修”成正确答案，而是解决旧结果中的主要失败模式：模型经常没有稳定看见目标对象，或没有按候选支撑面回答。

## 根因证据

基于 `vlm-semantic-eval-p4-replay-option-contract.json`，旧 raw response 在 p4 option-id contract 下离线 replay 后仍为：

- 60 条 QA。
- 语义正确：17/60，正确率 0.283333。
- 严格 exact match：0/60。

失败原因分布：

- `target_not_observed`：30 条。
- `destination_mismatch`：9 条。
- `relation_mismatch`：3 条。
- `relation_not_observed`：1 条。

这说明旧结果的主要瓶颈不是 JSON 解析或 answer option normalization，而是 VLM 输入侧仍不足以稳定引导模型定位目标与支撑面。

## 本阶段改动

`external_tools/run_vlm_controls.py` 的 VLM 请求构造现在会在每张图片前插入角色说明：

- `primary_frame`：主 RGB 图，用作目标位置的主要视觉证据。
- `target_crop`：目标 crop，用于确认目标对象，再回到主图选择支撑面或位置选项。
- `context_frame`：多帧 VLM 的辅助上下文帧。

trace 也会保存脱敏后的 `visual_prompt_payload` 和 `image_roles`，用于复盘每个 case 当时实际发送给 VLM 的结构化任务信息。这个 payload 保留问题、target label、answer options、support candidates 和 target crop 元信息，但不暴露机器 object id 给模型。

p6 进一步补充了 `visual_answer_option_strategy`：

- 如果 `answer_options` 非空且目标可见，必须选择且只选择一个合法 `answer_option_id`。
- target crop 只用于确认目标对象，支撑面/位置仍必须回到 primary RGB frame 判断。
- `room` 选项只能作为 fallback：没有更具体可见支撑面匹配时才使用。
- 如果目标不可见，返回 `target_not_observed`，不要选择 answer option。

这一步直接针对旧结果里的两个主要失败模式：`target_not_observed` 过多，以及模型自由生成目的地导致 `destination_mismatch`。

## 已更新 artifact

离线 replay 已用新 runner 重新生成：

- prediction：`handoffs/ai2thor-real-small/inputs/offline-controls/reruns/observation-aware-p4-target60/vlm-p4-replay-option-contract.jsonl`
- trace：`handoffs/ai2thor-real-small/outputs/offline-controls/observation-aware-p4-target60/traces/vlm-p4-replay-option-contract-trace.jsonl`
- semantic eval：`handoffs/ai2thor-real-small/outputs/diagnostics/vlm-semantic-eval-p4-replay-option-contract.json`

replay 输出保持：

- prediction count：60。
- trace count：60。
- missing case：0。
- prediction digest：`237b2d5004fd1bd045d012a62a644287678faa9bc38375419a6bf5d53be32cff`。
- semantic eval digest：`abae88f4f5a2f946eab6126344786c8a55a51539ed1e12c0abcedb773e3e7cb3`。

首条 replay trace 已包含：

- `image_roles=["primary_frame", "target_crop"]`
- `visual_prompt_payload.target={"label":"apple"}`
- `answer_option_count=3`
- `visual_answer_option_strategy.primary_rule="If answer_options is non-empty and the target is visible, choose exactly one allowed answer_option_id."`

## 验证

本阶段运行了：

```bash
python -m pytest tests/test_vlm_calibration.py tests/test_run_vlm_controls.py tests/test_run_vlm_detector_rgbd.py -q
```

结果：

- 51 passed。

## 阶段结论

这次改动提高的是下一轮真实 VLM-only 调用的输入可读性、选择约束和可审计性，还没有证明 VLM-only 成功率已经提升。旧 raw response 的 replay 仍然是 17/60，因为 replay 没有重新调用 VLM。

下一步要真正验证 VLM-only 是否提升，必须在明确允许外部 AI/network 的前提下，用 p4 option-id contract bundle 和新图片角色提示重新跑真实 VLM。成功判据建议为：

- 语义正确率显著高于当前 replay 的 17/60。
- `target_not_observed` 明显低于当前 30/60。
- trace 完整保存 raw response、normalized response、image roles 和 visual prompt payload。

DSG 侧当前 p7 observation-aware 诊断已达到 60/60，但它仍不是最终真实 detector-only 结论。真实结论需要下一轮 VLM-only 强 baseline 和真实 detector/RGB-D evidence-backed DSG 在同一 QA slice 上比较。
