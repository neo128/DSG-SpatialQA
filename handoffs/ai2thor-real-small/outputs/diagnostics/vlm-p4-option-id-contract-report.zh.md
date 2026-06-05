# VLM-only p4 优化阶段报告：answer option id 输出契约

## 目标

本阶段目标是先提高 VLM-only 的可答对上限，再继续比较 DSG。上一版 p3 已经把 answer option 的 gold 覆盖从 57/60 提升到 60/60，但外部 VLM 仍可能因为开放式自然语言输出而出现解析失败、关系漏填或目的地标签格式不稳定。

本阶段新增 p4 输入约束：每个可见空间 QA case 都提供稳定的 `answer_option_id`，并要求模型输出 `answer.answer_option_id`，再把对应选项的 `relation` 和 `destination_label` 复制到 `answer.current_location`。

## 新增 artifact

- VLM-only p4 request bundle:
  `handoffs/ai2thor-real-small/offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p4-option-id-contract.json`
- p4 answer option coverage report:
  `handoffs/ai2thor-real-small/outputs/diagnostics/vlm-p4-option-id-contract-coverage-report.json`

## 数据与结果

request bundle 摘要：

- case_count: 60
- answer_option_count: 193
- cases_with_answer_options: 60
- request_bundle_digest: `2c4f722b3f587b468358f2e32bc5614e01a6b5ca257241b9e7174fb159a7589e`

answer option 覆盖：

- covered_case_count: 60/60
- covered_case_rate: 1.0
- missing_case_count: 0
- report_digest: `83aa03022b310c923c8c4561e62da1e871892a122c580d48e7cf2f1960118eac`

泄漏审计：

- gold_answer occurrences: 0
- gold_evidence occurrences: 0
- visible_object_ids occurrences: 0
- support_candidate object_id occurrences: 0
- answer_option object_id occurrences: 0
- answer_option_response_schema cases: 60/60

## 原理

p4 不把 gold answer 或 oracle evidence 发给 VLM。它只把可见候选位置转成非 gold 的候选选项，例如：

```json
{
  "option_id": "locopt_001",
  "relation": "ON",
  "destination_label": "countertop"
}
```

模型输出时只需要选择 `answer.answer_option_id`。本地 runner 会在模型返回后，用同一个本地 request bundle 将该 option id 归一化成结构化 `current_location`。如果提供 `--normalization-frame-index`，runner 还能把可见标签映射回稳定本地 object id，用于后续 QA eval。

## 本轮新增容错

为了进一步降低 VLM-only 的非语义性失分，runner 现在支持三种合法 option id 入口：

- `answer.answer_option_id`
- 顶层 `answer_option_id`
- `answer_text` 中出现的合法 `locopt_###`
- 非 JSON/plain text 响应中唯一出现的合法 `locopt_###`
- 非 JSON/plain text 响应中的唯一合法序号/字母，例如 `Option 1`、`Choice 1`、`1`、`A`

只有当前 case 的 `answer_options` 中存在的 option id 会被接受；不会接受任意自由文本，也不会把隐藏 object id 发送给模型。prompt 侧也会显式包含 `answer_option_response_schema`，让外部 VLM 清楚知道必须从 `allowed_answer_option_ids` 中选择。

如果模型完全没有按 JSON schema 返回、只返回 `locopt_001`、`Option 1` 或 `A` 这类纯文本，runner 会先检查该 id 或序号是否能唯一映射到当前 case 的 `allowed_answer_option_ids`。满足条件时，它会恢复出一个带 `recovered_from="plain_text_answer_option"` 的结构化响应，再进入同一套本地归一化流程。多个冲突匹配、越界序号或没有合法选项时不会恢复。

## 当前结论

p4 已经解决了 VLM-only 输入侧的两个主要问题：

1. 正确答案候选覆盖达到 60/60。
2. 输出格式从开放式空间描述收敛为稳定 option id，降低解析失败概率。
3. 即使模型把合法 option id 写在顶层字段、`answer_text`、纯文本响应，或写成 `Option 1` / `A`，本地归一化也能恢复成可评测的 `current_location`。

但这还不是 VLM-only 实测成功率提升。下一步必须用该 p4 bundle 重跑真实外部 VLM-only prediction JSONL，然后重新生成 semantic eval 和 DSG-vs-VLM delta。

推荐下一步外部重跑输入：

```bash
python external_tools/run_vlm_controls.py \
  --request-bundle handoffs/ai2thor-real-small/offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p4-option-id-contract.json \
  --normalization-frame-index handoffs/ai2thor-real-small/inputs/traces/vlm-frame-index-coverage-p3.jsonl \
  --source-kind vlm \
  --model qwen3.7-plus \
  --api-key-env DSG_SPATIALQA_DASHSCOPE_API_KEY \
  --allow-network \
  --output handoffs/ai2thor-real-small/inputs/offline-controls/reruns/vlm-p4-option-id-contract.jsonl \
  --trace-output handoffs/ai2thor-real-small/inputs/offline-controls/reruns/vlm-p4-option-id-contract-trace.jsonl
```

注意：本阶段未调用外部 VLM，不宣称 VLM-only 实测准确率已经提升。
