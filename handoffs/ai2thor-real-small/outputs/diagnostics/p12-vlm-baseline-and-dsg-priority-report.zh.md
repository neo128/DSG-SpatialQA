# P12 VLM baseline 强化与 DSG 后续优先级报告

## 目标

本阶段继续执行“先提高 VLM-only 成功率，再优化 DSG”的目标。所有改动均为本地
代码、评估和 artifact 生成；没有调用外部 VLM/LLM API。

## VLM-only 本地提升

本轮修复了 VLM 语义评估中的两个本地问题：

1. 结构化 VLM answer 中 `current_location.dst_label` / `destination_label`
   没有被语义评估读取。
2. `dining room` 没有被视为 room-level destination，导致
   `IN_ROOM/room` 的正确回答被判为 destination mismatch。

修复后重新评估当前已保存的真实 VLM prediction：

| 系统 | 旧语义正确数 | 新语义正确数 | 新语义正确率 |
| --- | ---: | ---: | ---: |
| VLM-only | 17/60 | 22/60 | 0.366667 |
| Multi-frame VLM | 17/60 基线未单列 | 25/60 | 0.416667 |

新增报告：

- `vlm-semantic-eval-p12-dst-label-room-independent.json`
- `multi-frame-vlm-semantic-eval-p12-dst-label-room-independent.json`
- `vlm-semantic-delta-p12-vs-p5-canonicalized-independent.json`

VLM-only p12 相比 p5 canonicalized：

- +5 semantic matches
- paired wins: 5
- paired losses: 0
- digest: `c7c5255aa797318137e8fb14b9a8162e92b1ed083103239171cf76a6140d9c51`

## DSG 本地安全修复

本轮也修复了 DSG 查询层的一个安全问题：当支撑/家具类目标同时存在
`IN_ROOM` 和过度生成的 `ON` containment edge 时，`object_location` 优先返回
room-level location，避免把椅子等目标回答成 `ON chair`。

新增测试覆盖：

- 普通物体仍优先使用 detector current location。
- 支撑/家具类目标在同 step 有 room edge 时优先返回 `IN_ROOM`。

当前 p4 predicted graph 重新跑 GraphTool 后，DSG 分数未变化：

| 系统 | 语义正确数 | 语义正确率 |
| --- | ---: | ---: |
| DSG p10/p13 GraphTool | 23/60 | 0.383333 |

说明：当前主要瓶颈不是查询优先级，而是 predicted graph 缺 support/object evidence。

## DSG vs 强化后的 VLM baseline

新增 delta：

- `dsg-candidate-semantic-delta-vs-vlm-p12-dst-label-room-independent.json`
- `dsg-candidate-semantic-delta-vs-multi-frame-vlm-p12-dst-label-room-independent.json`

结果：

| 对比 | DSG | Baseline | Delta |
| --- | ---: | ---: | ---: |
| DSG p13 vs VLM-only p12 | 23/60 | 22/60 | +1 |
| DSG p13 vs Multi-frame VLM p12 | 23/60 | 25/60 | -2 |

阶段结论：

- DSG 仍略高于 VLM-only，但优势已经很薄。
- DSG 当前不优于 multi-frame VLM。
- 因此不能声称 DSG 已经稳定优于视频记忆/VLM。

## 剩余根因

VLM p12 剩余失败：

- VLM-only：`target_not_observed=30`、`destination_mismatch=4`、
  `relation_not_observed=1`、`relation_mismatch=3`。
- Multi-frame VLM：`target_not_observed=27`、`destination_mismatch=5`、
  `relation_mismatch=3`。

DSG p10/p13 剩余失败：

- 大量 `Object not found`，说明 detector graph 缺目标对象。
- 大量 `relation_mismatch`，通常是 GraphTool 只能回退到 `IN_ROOM`，
  缺 `ON support` 关系。
- support gap 报告显示：`target_missing=15`、`support_missing=14`、
  `support_present_relation_missing=3`。
- 需要优先补采/补检的 support labels：
  `coffeetable`、`countertop`、`dresser`、`handtowelholder`、`shelf`。

## 下一步

1. 用 p5 visual-contract request bundle 重跑真实 VLM-only 和 multi-frame VLM。
2. 用 support gap 生成 detector/RGB-D 补采集任务，优先覆盖上述 5 类 support。
3. 重新构建 observation-sequence-backed predicted DSG。
4. 再跑 DSG GraphTool、VLM-only、multi-frame VLM 的同 QA 对比。

只有当新的 detector-backed DSG 在强化后的 VLM baselines 上仍有稳定正 delta，
才可以下更强结论。
