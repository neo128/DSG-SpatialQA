# VLM-only p2 Answer Options 输入优化报告

## 目标

本阶段目标是提高下一轮 VLM-only 的成功率。上一轮 p5 语义评估显示，VLM-only 在 `observation-aware-p4-target60` 上为 `17/60`，主要失败来自：

- `target_not_observed`: 30 条
- `destination_mismatch`: 9 条
- `relation_mismatch`: 3 条
- `relation_not_observed`: 1 条

这说明 VLM-only 的主要瓶颈不是评估 parser，而是模型在单帧中经常选择 unknown，或者把目标放到错误支撑物上。

## 方法

在不调用外部模型、不泄露 gold answer/gold evidence 的前提下，基于已有 p1 `support_candidates + target_crop` request bundle 生成 p2 answer-options bundle。

每个 case 的 `answer_options` 只包含：

- `option_id`
- `relation`
- `destination_label`
- `source`

其中 support candidates 生成 `ON/INSIDE <support label>`，并加入保守的 `IN_ROOM room` fallback。支撑物机器 id 会被清理，不进入外部 VLM prompt。

## 新增 Artifact

- p2 request bundle:
  `handoffs/ai2thor-real-small/offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p2-support-crop-options.json`
- digest:
  `cf762506fa3ad5f47d5dfcc0fc052edfe51251437bda4a6b777d88c62c15401d`

## 覆盖统计

| 指标 | 数值 |
| --- | ---: |
| case_count | 60 |
| cases_with_answer_options | 60 |
| answer_option_count | 190 |
| min_options_per_case | 1 |
| max_options_per_case | 6 |
| cases_with_target_crop | 39 |
| cases_with_support_candidates | 53 |
| cases_with_room_option | 60 |

无泄漏检查：

- `gold_answer`: 0
- `gold_evidence`: 0
- `visible_object_ids`: 0
- `support_candidate.object_id`: 0

## 预期收益

p2 的作用是降低 VLM 自由文本回答的不稳定性，让模型在可见支撑物候选中选择，而不是输出 `unknown`、`current frame`、`current view` 或错误泛化位置。

这不能保证 VLM-only 一定超过当前 `17/60`，但它直接针对了当前最大失败族：`target_not_observed` 和 `destination_mismatch`。

## 推荐重跑命令

```bash
DSG_SPATIALQA_DASHSCOPE_API_KEY=... \
python external_tools/run_vlm_controls.py \
  --request-bundle handoffs/ai2thor-real-small/offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p2-support-crop-options.json \
  --source-kind vlm \
  --model qwen3.7-plus \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --allow-network \
  --normalization-frame-index handoffs/ai2thor-real-small/inputs/traces/vlm-frame-index-coverage-p3.jsonl \
  --output handoffs/ai2thor-real-small/inputs/offline-controls/reruns/observation-aware-p4-target60/vlm-p2-support-crop-options.jsonl \
  --trace-output handoffs/ai2thor-real-small/inputs/traces/reruns/vlm-p2-support-crop-options-trace.jsonl
```

## 注意

本阶段只完成 VLM-only 输入优化和本地 artifact 准备，没有调用外部 VLM。下一步需要用项目级 API key 重跑 VLM-only p2，然后重新生成 semantic eval 和 DSG-vs-VLM delta。
