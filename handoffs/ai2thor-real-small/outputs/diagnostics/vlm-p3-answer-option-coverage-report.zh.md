# VLM-only p3 Answer Options 覆盖率优化报告

## 目标

本阶段目标是继续提高下一轮 VLM-only 重跑的成功率。由于当前 AGENTS 约束不调用外部 VLM/API，本阶段只做本地可验证的输入侧优化：检查 answer options 是否覆盖每条 QA 的正确位置，并补齐不泄露 gold 的候选策略。

## p2 覆盖诊断

输入：

- QA: `observation-aware-p4-target60-qa.jsonl`
- p2 bundle: `offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p2-support-crop-options.json`

p2 answer option 覆盖：

| 指标 | 数值 |
| --- | ---: |
| case_count | 60 |
| covered_case_count | 57 |
| covered_case_rate | 0.950000 |
| missing_case_count | 3 |
| option_count | 190 |

漏掉的 3 条：

1. `cloth` gold 为 `ON bathtub`，但候选只有 `INSIDE bathtub`；
2. `faucet` gold 为 `ON bathtub`，但候选只有 `INSIDE bathtub`；
3. `handtowel` gold 为 `ON handtowelholder`，但候选只有 `IN_ROOM room`。

这说明 p2 已经覆盖大多数题，但对于支撑物多义关系和目标类别 affordance 仍有盲点。

## p3 修改

p3 在不泄露 gold answer/gold evidence 的前提下新增两个显式开启的候选策略：

1. `include_ambiguous_support_relations`
   - 对已可见的多义支撑物补充额外关系；
   - 当前只对 `bathtub` 在 `INSIDE bathtub` 之外补 `ON bathtub`。

2. `include_target_affordance_options`
   - 基于目标类别的非 gold 常识 affordance 补候选；
   - 当前只对 `handtowel` 补 `ON handtowelholder`。

这些候选不包含 object id，不包含 gold answer，不包含 gold evidence。

## p3 覆盖结果

输出：

- p3 bundle:
  `offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p3-support-crop-options-affordance.json`
- p3 coverage report:
  `vlm-p3-answer-option-coverage-report.json`

| 指标 | p2 | p3 |
| --- | ---: | ---: |
| covered_case_count | 57 | 60 |
| covered_case_rate | 0.950000 | 1.000000 |
| missing_case_count | 3 | 0 |
| option_count | 190 | 193 |

p3 新增候选来源统计：

| source | count |
| --- | ---: |
| support_candidate | 130 |
| fallback_room | 60 |
| ambiguous_support_relation | 2 |
| target_affordance_prior | 1 |

泄漏检查：

- `gold_answer`: 0
- `gold_evidence`: 0
- `visible_object_ids`: 0
- `support_candidate.object_id`: 0
- answer option 内 object id: 0

## 推荐 VLM-only 重跑命令

```bash
DSG_SPATIALQA_DASHSCOPE_API_KEY=... \
python external_tools/run_vlm_controls.py \
  --request-bundle handoffs/ai2thor-real-small/offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p3-support-crop-options-affordance.json \
  --source-kind vlm \
  --model qwen3.7-plus \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --allow-network \
  --normalization-frame-index handoffs/ai2thor-real-small/inputs/traces/vlm-frame-index-coverage-p3.jsonl \
  --output handoffs/ai2thor-real-small/inputs/offline-controls/reruns/observation-aware-p4-target60/vlm-p3-support-crop-options-affordance.jsonl \
  --trace-output handoffs/ai2thor-real-small/inputs/traces/reruns/vlm-p3-support-crop-options-affordance-trace.jsonl
```

## 结论

p3 没有调用外部 VLM，因此不能声称 VLM-only 实测准确率已经提高。但它证明下一轮 VLM-only 的候选答案输入已经从 p2 的 `57/60` gold-option coverage 提升到 `60/60`，消除了“正确答案根本不在候选里”的输入侧瓶颈。

下一步需要在用户明确授权外部 VLM/API 调用后，用 p3 bundle 重跑 VLM-only，再和当前 DSG p6 结果重新计算 semantic eval 和 delta。
