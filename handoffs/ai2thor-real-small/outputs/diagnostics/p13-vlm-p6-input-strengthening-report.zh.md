# P13 VLM-only 输入强化报告

## 目标

本阶段继续执行“先提高 VLM-only 成功率，再优化 DSG”的路线。本轮没有调用外部
VLM/LLM API，只处理本地 request bundle、runner prompt 和诊断报告。

## 本轮发现

VLM-only p12 的剩余失败主要是 `target_not_observed`。进一步检查 p5
visual-contract 请求包后发现，所有样本都有 answer options，但 target crop 覆盖仍
不足：

| 分组 | p5 target crop 覆盖 |
| --- | ---: |
| correct | 21/22 |
| destination_mismatch | 2/4 |
| relation_mismatch | 1/3 |
| relation_not_observed | 1/1 |
| target_not_observed | 14/30 |

其中一类局部问题是 detector bbox 使用了 0-1000 坐标系，而裁剪代码直接按原始像素
坐标解释，导致部分 bbox 被判定为出界，target crop 没有写入。

## 已完成改动

1. `vlm_target_crop_request_bundle()` 现在会识别明显的 0-1000 detector bbox，并将其
   缩放到实际 PPM 图像尺寸后再裁剪。
2. VLM runner 的 `visual_decision_checklist` 增加了 crop-first 可见性约束：
   当 `target_crop` 可用时，先用 crop 确认小目标或部分可见目标，再决定是否返回
   `target_not_observed`。
3. 重新生成了 p6 VLM-only request bundle：
   `handoffs/ai2thor-real-small/offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p6-rescaled-bbox-visual-contract.json`
4. 新增结构化诊断：
   `handoffs/ai2thor-real-small/outputs/diagnostics/vlm-p6-rescaled-bbox-input-report.json`

## 输入覆盖变化

| 指标 | p5 | p6 |
| --- | ---: | ---: |
| case_count | 60 | 60 |
| cases_with_target_crop | 39 | 41 |
| target_crop_delta | - | +2 |

p6 request bundle digest:
`b8b5acd79cb100e5c6eb6ff3870ac292398998e8d3289bb63e40ae1319fbf33c`

诊断报告 digest:
`fc0b89998ec67aaaea3bcb1423cc58bd1269c2efcb5d7e71bf49fec7040db47a`

## Gold 泄露检查

p6 bundle 未包含 `gold_answer` 或 `gold_evidence`。本轮强化的是可视输入和输出约束，
不是把 oracle answer 暴露给外部 VLM。

## 阶段结论

本轮提高的是 VLM-only 的可执行输入质量，还不能声称 VLM-only 准确率已经提高。
下一步需要用 p6 request bundle 重跑真实 VLM-only / multi-frame VLM prediction，再用
同一套 semantic eval 复评。

当前 DSG 结论仍保持不变：DSG p13 对 VLM-only p12 只领先 1 题，对 multi-frame VLM
p12 落后 2 题，因此尚不能证明 DSG 稳定优于 VLM / 视频记忆。

## 下一步执行入口

在外部 VLM 调用被明确授权后，优先运行 p6：

```bash
python external_tools/run_vlm_controls.py \
  --request-bundle handoffs/ai2thor-real-small/offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p6-rescaled-bbox-visual-contract.json \
  --source-kind vlm \
  --normalization-frame-index handoffs/ai2thor-real-small/inputs/traces/vlm-frame-index-coverage-p3.jsonl \
  --output handoffs/ai2thor-real-small/inputs/offline-controls/reruns/observation-aware-p4-target60-p6/vlm.jsonl \
  --trace-output handoffs/ai2thor-real-small/inputs/offline-controls/reruns/observation-aware-p4-target60-p6/vlm-trace.jsonl \
  --api-key-env DSG_SPATIALQA_DASHSCOPE_API_KEY \
  --allow-network
```

重跑后再生成：

- VLM-only p6 semantic eval；
- multi-frame VLM p6 semantic eval；
- DSG p13 vs VLM p6 delta；
- DSG p13 vs multi-frame VLM p6 delta。
