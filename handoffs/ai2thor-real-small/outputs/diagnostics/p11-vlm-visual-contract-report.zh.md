# P11 VLM-only 视觉合约优化报告

## 目标

本阶段先提高 VLM-only baseline 的输入质量，再继续优化 DSG。当前不调用外部
VLM API，不修改系统环境变量，也不声称 VLM 分数已经提升。

## 根因

直接从 offline-control manifest 生成的 request bundle 可以提供 60 条 QA，
但没有 `answer_options` 和 `target_crop`。这会让 VLM-only 在 object-location
QA 上直接生成自由文本或猜测位置，难以稳定映射到评估答案。

已有的 p4 option-contract bundle 证明：VLM 输入必须显式提供非 gold 的可见支撑候选、
目标 crop 和合法 answer option id，才能让模型输出结构化、可归一化的答案。

## 本轮改动

- 在 `external_tools/run_vlm_controls.py` 的 VLM prompt payload 中加入
  `visual_decision_checklist`。
- checklist 要求模型按顺序执行：
  1. 判断目标是否在 primary frame 或 target crop 中可见；
  2. 目标可见且存在 answer options 时，只选择一个视觉支持的 option；
  3. 在 `observability` 或 `reasoning_summary` 中说明证据来自哪个 image role。
- 在 `scripts/eval_vlm_calibration.py` 增加
  `--enhanced-vlm-request-bundle-output`，把 support candidate、target crop 和
  answer option 三个已有 enrichment 串成一个本地命令。

## 新生成的本地 VLM 输入

输出 bundle：

`handoffs/ai2thor-real-small/offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p5-visual-contract.json`

digest：

`e163a8eacec273949f34c83386eb5f505cc717e2b988267e7f1389a51526dcc8`

统计：

| 项目 | 数量 |
| --- | ---: |
| QA cases | 60 |
| cases with answer options | 60 |
| cases with support candidates | 53 |
| cases with target crop | 39 |
| target crop files | 39 |

gold 隔离检查：

- `gold_answer`: absent
- `gold_evidence`: absent
- `expected_answer`: absent
- `oracle_answer`: absent

说明：prompt 中允许出现说明性 `non_gold_rule`，但不包含 gold answer 或 gold evidence
字段。

## 当前结论

本轮提升的是 VLM-only 的输入和输出合约，不是模型结果。下一步需要在用户重新授权外部
VLM 后，用该 p5 bundle 重跑 `vlm` 和 `multi_frame_vlm`，再和 DSG p10 做正式语义对比。

推荐下一步命令使用项目作用域 key：

```bash
DSG_SPATIALQA_DASHSCOPE_API_KEY=... \
python external_tools/run_vlm_controls.py \
  --request-bundle handoffs/ai2thor-real-small/offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p5-visual-contract.json \
  --source-kind vlm \
  --model qwen3.7-plus \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --allow-network \
  --normalization-frame-index handoffs/ai2thor-real-small/inputs/traces/vlm-frame-index-coverage-p3.jsonl \
  --output handoffs/ai2thor-real-small/inputs/offline-controls/reruns/observation-aware-p4-target60/vlm-p5-visual-contract.jsonl \
  --trace-output handoffs/ai2thor-real-small/inputs/traces/reruns/vlm-p5-visual-contract-trace.jsonl
```

在未重跑真实 VLM 前，不能得出“VLM-only 已提升”或“DSG 已最终优于 VLM”的结论。
