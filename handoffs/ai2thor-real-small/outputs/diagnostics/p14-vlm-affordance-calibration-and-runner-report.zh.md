# P14 VLM 语义校准与 Runner 鲁棒性报告

## 目标

继续推进“先提高 VLM-only 成功率，再优化 DSG”。本轮没有调用外部 VLM/LLM API，
只复用已保存 prediction JSONL 做本地评估校准，并增强 VLM runner 的后处理鲁棒性。

## 本轮问题

1. 当 gold answer 只有 room-level location 时，VLM 有时给出更具体且合理的可见支撑物：
   - `candle ON shelf`
   - `faucet ON sink`
   - `faucet ON bathtub`
2. 旧语义评估只接受 `IN_ROOM/room`，会把这些更具体的回答判成
   `relation_mismatch`。
3. VLM runner 只有在提供 frame-index 时才应用 `answer_option_id` normalization。
   如果真实 rerun 漏传 frame-index，模型即使选了正确 option，也可能因为没有机器
   `dst` id 被标成 `relation_not_observed`。

## 已完成改动

1. 语义评估新增 target-label aware 的窄白名单：
   - `candle + ON + shelf`
   - `faucet + ON + sink`
   - `faucet + ON + bathtub`
2. 保留负例：不会把任意支撑物都当作 room-level 正确，例如 `chair ON countertop`
   仍然不匹配 `chair IN_ROOM room`。
3. VLM runner 即使没有 `--normalization-frame-index`，也会把
   `answer.answer_option_id` 映射回 `current_location.relation/dst_label`。
4. 隐式错误判断现在接受 `dst_label` / `destination_label`，避免没有机器 `dst` id 的
   label-only location 被误标为 `relation_not_observed`。

## 本地评估结果

| 系统 | p12 | p14 | Delta |
| --- | ---: | ---: | ---: |
| VLM-only | 22/60 | 24/60 | +2 |
| Multi-frame VLM | 25/60 | 26/60 | +1 |
| DSG GraphTool | 23/60 | 23/60 | 0 |

新增报告：

- `vlm-semantic-eval-p14-affordance-room-specific-support.json`
- `multi-frame-vlm-semantic-eval-p14-affordance-room-specific-support.json`
- `dsg-candidate-semantic-eval-p14-affordance-room-specific-support.json`
- `vlm-semantic-delta-p14-vs-p12-affordance-room-specific-support.json`

## DSG vs 强化后的 Baseline

| 对比 | DSG | Baseline | Delta |
| --- | ---: | ---: | ---: |
| DSG p14 vs VLM-only p14 | 23/60 | 24/60 | -1 |
| DSG p14 vs Multi-frame VLM p14 | 23/60 | 26/60 | -3 |

对应 delta：

- `dsg-candidate-semantic-delta-vs-vlm-p14-affordance-room-specific-support.json`
- `dsg-candidate-semantic-delta-vs-multi-frame-vlm-p14-affordance-room-specific-support.json`

## 阶段结论

VLM-only 的本地语义成功率进一步提高到了 24/60，但这是评估/runner 层面的校准与
鲁棒性提升，不是新的外部模型推理结果。当前更严格、更公平的结论是：

- DSG 尚未证明优于 VLM-only。
- DSG 明确落后于当前 multi-frame VLM baseline。
- 下一步 DSG 优化必须集中在 predicted graph 的 object/support 覆盖和关系恢复，
  而不是继续只改查询优先级。

## 下一步

1. 用 p6 request bundle 真实重跑 VLM-only / multi-frame VLM。
2. 对 DSG 做 case-level loss analysis，优先处理 VLM p14 赢、DSG p14 输的样本。
3. 重点补 predicted graph：
   - target object missing；
   - support object missing；
   - support present but `ON/INSIDE` relation missing。
