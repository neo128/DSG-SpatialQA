# VLM-only p4 离线 trace replay 报告

## 目的

本阶段不调用外部 VLM，只使用已经保存的本地 `raw_response` trace，验证最近新增的 VLM-only 归一化和 option-id 容错能否直接挽回旧结果。

## 输入

- p4 request bundle:
  `handoffs/ai2thor-real-small/offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p4-option-id-contract.json`
- 旧 VLM trace:
  `handoffs/ai2thor-real-small/outputs/offline-controls/observation-aware-p4-target60/traces/vlm-trace.jsonl`
- frame index:
  `handoffs/ai2thor-real-small/inputs/traces/vlm-frame-index-coverage-p3.jsonl`
- QA:
  `handoffs/ai2thor-real-small/inputs/traces/observation-aware-p4-target60-qa.jsonl`

## 输出

- replay prediction JSONL:
  `handoffs/ai2thor-real-small/inputs/offline-controls/reruns/observation-aware-p4-target60/vlm-p4-replay-option-contract.jsonl`
- replay trace:
  `handoffs/ai2thor-real-small/outputs/offline-controls/observation-aware-p4-target60/traces/vlm-p4-replay-option-contract-trace.jsonl`
- semantic eval:
  `handoffs/ai2thor-real-small/outputs/diagnostics/vlm-semantic-eval-p4-replay-option-contract.json`

## 结果

Replay 覆盖：

- prediction_count: 60
- trace_count: 60
- missing_case_count: 0
- missing_raw_response_case_count: 0
- prediction_digest: `237b2d5004fd1bd045d012a62a644287678faa9bc38375419a6bf5d53be32cff`

Semantic eval：

- case_count: 60
- matched_prediction_count: 60
- semantic_match_count: 17
- semantic_match_rate: 0.2833333333333333
- strict_exact_match_count: 0
- report_digest: `abae88f4f5a2f946eab6126344786c8a55a51539ed1e12c0abcedb773e3e7cb3`

## 结论

离线 replay 没有把旧 VLM-only 结果从 17/60 拉高。这说明旧 `raw_response` 的主要失败不只是 JSON/格式解析失败，而是模型当时没有选择正确空间位置或没有看到足够证据。

因此，最近新增的 p4 option-id contract、plain-text/ordinal 容错是必要的基础设施，但它们不能凭空修复旧模型回答。要真正验证 VLM-only 成功率是否提高，下一步必须用 p4 request bundle 重新跑外部 VLM-only，让模型在生成时直接看到 answer options 和响应契约。

推荐下一步仍是：

```bash
python external_tools/run_vlm_controls.py \
  --request-bundle handoffs/ai2thor-real-small/offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p4-option-id-contract.json \
  --normalization-frame-index handoffs/ai2thor-real-small/inputs/traces/vlm-frame-index-coverage-p3.jsonl \
  --source-kind vlm \
  --model qwen3.7-plus \
  --api-key-env DSG_SPATIALQA_DASHSCOPE_API_KEY \
  --allow-network \
  --output handoffs/ai2thor-real-small/inputs/offline-controls/reruns/observation-aware-p4-target60/vlm-p4-option-id-contract.jsonl \
  --trace-output handoffs/ai2thor-real-small/outputs/offline-controls/observation-aware-p4-target60/traces/vlm-p4-option-id-contract-trace.jsonl
```

当前 AGENTS 约束下，本阶段没有调用外部网络，也不声称 VLM-only 实测成功率已经提升。
