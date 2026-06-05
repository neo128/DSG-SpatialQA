# P15 DSG Coverage-Merged 支撑关系恢复报告

## 目标

继续推进“提高 VLM-only 成功率，再优化 DSG”。P14 已把已保存的 VLM-only
语义成功率校准到 24/60，把 multi-frame VLM 校准到 26/60，但 DSG GraphTool
仍只有 23/60。本轮重点排查 DSG 为什么落后。

本轮没有调用外部 VLM/LLM API，也没有启动仿真器或 detector。

## 根因

DSG 失败的主因不是目标物完全缺失，而是 predicted graph 缺少可查询的支撑物节点和
`ON` 关系边。旧 independent p4 predicted graph 的结构特征是：

| 图 | 节点数 | 边数 | `ON` 边 | countertop | dresser | shelf | sink |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| independent p4 | 111 | 116 | 3 | 0 | 0 | 0 | 0 |
| coverage-merged p2 | 904 | 2996 | 458 | 6 | 1 | 7 | 1 |
| coverage-merged p3 | 1051 | 3459 | 548 | 6 | 1 | 8 | 1 |

也就是说，旧 DSG candidate 把很多问题退化为 `IN_ROOM room`，因为图里没有足够的
`countertop/dresser/shelf/sink` 支撑节点和 `ON` 边。

## 本轮修复

1. `episode_metadata_coverage_detector_jsonl` 和 segmentation detector export 现在会在
   detector record 顶层写入 `episode_id` 和 `scene_id`。
2. VLM target-crop 索引对旧 metadata-only detector record 增加兼容：
   如果顶层没有 `episode_id/scene_id`，会从 `metadata` 回退读取。
3. 生成了 schema-fixed 的 coverage detector JSONL 副本：
   - `detector-rgbd-coverage-ai2thor-visible-p2-schema-fixed-p15.jsonl`
   - `detector-rgbd-coverage-ai2thor-visible-p3-offset-schema-fixed-p15.jsonl`
4. 两个 fixed JSONL 均可导入为 SceneObservation sequence：
   - p2 fixed：48 frames，493 visible object observations
   - p3 fixed：14 frames，90 visible object observations

## 本地评估

同一套 `observation-aware-p4-target60-qa.jsonl`，60 条 QA：

| 系统 | 成功数 | 成功率 |
| --- | ---: | ---: |
| VLM-only p14 | 24/60 | 40.0% |
| Multi-frame VLM p14 | 26/60 | 43.3% |
| DSG independent p14 | 23/60 | 38.3% |
| DSG coverage-merged p2 p15 | 43/60 | 71.7% |
| DSG coverage-merged p3 p15 | 59/60 | 98.3% |

对应 delta：

| 对比 | Delta |
| --- | ---: |
| DSG p15 p2 vs VLM-only p14 | +19 |
| DSG p15 p2 vs Multi-frame VLM p14 | +17 |
| DSG p15 p3 vs VLM-only p14 | +35 |
| DSG p15 p3 vs Multi-frame VLM p14 | +33 |

P15 p3 唯一失败样本是：

- `diningtable_02_27_00_02_01_42`
- gold：`diningtable ON chair`
- prediction：`diningtable IN_ROOM room`
- failure：`relation_mismatch`

## 阶段结论

P15 证明了一个关键点：DSG 失败不是 GraphTool 天然不行，而是 predicted graph 的
support-relation evidence 没有进入图。一旦 coverage-merged graph 具备足够的支撑物节点
和 `ON` 边，DSG 在同一 QA slice 上可以明显超过 VLM-only 和 multi-frame VLM。

但这还不是最终研究结论。原因是 p15 使用的是 AI2-THOR metadata-backed coverage，
更适合作为诊断上界或 oracle-like perception upper bound。要得出“真实 DSG 优于 VLM /
视频记忆”的结论，还需要把同样的支撑关系恢复能力迁移到真实外部 detector/RGB-D
observation sequence，并通过 readiness gate。

## 下一步

1. 用真实 detector/RGB-D 输出重建 p15 这种 support-rich observation sequence。
2. 要求外部 detector 输出支撑物类别和 3D bbox：
   `countertop/dresser/shelf/sink/bathtub/handtowelholder/chair/table`。
3. 在 predicted graph report 中新增或检查 support node recall、`ON` edge count、目标到支撑物
   relation coverage。
4. 重新跑真实 DSG candidate，并只在 external detector-backed graph 上报告研究结论。
