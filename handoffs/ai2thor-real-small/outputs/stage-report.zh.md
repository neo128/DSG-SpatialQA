# ai2thor-real-small 阶段性实验报告

## 1. 报告要求

后续每次真实或半真实实验都需要给出中文阶段性实验报告。报告至少包含：

- 实验目标：本阶段要验证什么，哪些问题暂不验证。
- 实验过程：数据如何采集、预测如何生成、评估如何运行、哪些 artifact 被保存。
- 方法原理：解释 DSG、GraphTool、VLM/LLM control、QA eval、graph eval、error attribution 的作用。
- 实验数据：列出 episode、frame、QA、prediction、graph、dashboard、readiness 等关键数量和路径。
- 指标解释：不仅列数值，还要说明指标含义和为什么会高或低。
- 阶段结论：明确区分流程是否完整、结果是否达标、下一步要补什么。

本报告覆盖当前 `ai2thor-real-small` 小规模真实包和 `candidate_v2` 诊断侧包。

## 2. 实验目标

本阶段目标是运行并审计一个小规模真实实验包：

- 使用真实 AI2-THOR 采集 artifact，而不是 mock episode。
- 建立 5 个 episode、60 条 QA、4 组外部/离线 control prediction。
- 构建 observation-sequence-backed predicted DSG。
- 运行 `graph_tool` candidate、QA eval、QA delta、graph eval、error attribution、dashboard、readiness report 和 experiment record。
- 针对原始结果很差的问题，保存完整过程链路，并构建一个不覆盖原始结果的 `candidate_v2` 诊断版本。

本阶段不把 `candidate_v2` 当成最终模型结果。它只用于定位原始 predicted DSG 为什么无法回答多数 QA。

## 3. 实验过程

### 3.1 真实数据与 benchmark

实验包目录为 `handoffs/ai2thor-real-small/`。当前包含：

- 5 个 AI2-THOR episode JSONL：`inputs/episodes/ai2thor-real-small-episode-001.jsonl` 到 `005.jsonl`。
- 50 帧 frame asset：每个 episode 10 帧，包含 RGB、depth、segmentation。
- 60 条 benchmark QA：`inputs/qa.jsonl`。
- 5 个 oracle graph：`outputs/benchmark/graphs/*-oracle-graph.json`。
- 一个 combined oracle graph：`outputs/oracle/oracle-graph.json`。

benchmark QA 是从 oracle graph 生成的，因此 QA 覆盖了完整场景里的对象和关系。这一点对评估很重要：oracle 知道的对象不一定被 RGB-D/detector observation sequence 实际观测到。

### 3.2 外部 VLM/LLM control

本阶段已有 4 组 control prediction：

- `vlm`
- `multi_frame_vlm`
- `caption_memory`
- `graph_text`

每组都导入为标准 QA prediction JSONL，并与同一份 `inputs/qa.jsonl` 对齐。总导入 prediction 数为 240，即 4 个 source 各 60 条。

相关路径：

- `inputs/offline-controls/vlm.jsonl`
- `inputs/offline-controls/multi_frame_vlm.jsonl`
- `inputs/offline-controls/caption_memory.jsonl`
- `inputs/offline-controls/graph_text.jsonl`
- `outputs/offline-controls/imports/*/predictions.jsonl`
- `outputs/offline-controls/offline-control-matrix.json`

早期 VLM/LLM 运行保存了最终 prediction 和 raw response，但没有保存逐 case 的完整 request payload。因此本阶段补充了 trace，将 QA、图片、prediction、raw response 关联起来；prompt/context 字段标记为从已保存输入重建，而不是伪装成原始请求体。

### 3.3 observation-backed predicted DSG

predicted DSG 来自 detector/RGB-D observation sequence，而不是 oracle graph。相关路径：

- detector observation 输入：`inputs/predicted-dsg/detector-rgbd.jsonl`
- observation sequence：`outputs/predicted-dsg/detector-observations.json`
- 原始 predicted graph：`outputs/predicted-dsg/predicted-graph.json`
- 原始 graph report：`outputs/predicted-dsg/predicted-graph-report.json`

原始 predicted graph 只包含 38 个对象，而 oracle graph 包含 176 个对象。也就是说，许多 QA 目标对象从一开始就不在 predicted graph 里。

### 3.4 过程 trace 补全

为了让实验过程可审计，本阶段新增 trace artifact：

- `inputs/traces/frame-index.jsonl`：50 条 frame 记录，关联 RGB/depth/segmentation 路径和 digest。
- `inputs/traces/raw-response-index.jsonl`：8 条 raw response 索引。
- `inputs/traces/vlm-interactions.jsonl`：240 条 VLM/LLM interaction 记录。
- `inputs/traces/qa-trace.jsonl`：60 条 QA 逐例 trace，关联 gold、prediction、图片、raw response、error attribution。
- `inputs/traces/visibility-aligned-qa.jsonl`：13 条目标对象都存在于 predicted graph 的 QA 子集。
- `inputs/traces/trace-readiness.json`：trace readiness gate。

trace readiness 当前为 `ready: true`。关键检查结果：

- frame index：50/50。
- frame assets present：50/50。
- QA trace：60/60。
- VLM interactions：240/240。
- raw response records：8，满足至少 4 条的要求。
- visibility-aligned QA：13/13。

## 4. 方法原理

### 4.1 Oracle graph 与 predicted graph

Oracle graph 使用 simulator metadata 构建，代表完整场景真值。Predicted graph 使用 observation sequence 构建，代表实际观测和 detector/RGB-D 能恢复出的图。

二者差异正是实验要测的东西：如果 predicted DSG 缺对象、缺关系或状态不完整，GraphTool 在 QA 上会失败。

### 4.2 GraphTool QA

GraphTool 不直接看图片，而是在 DSG 上查询对象位置、关系、状态和时间线。它的优势应该来自结构化记忆；它的弱点是高度依赖图构建质量。

本实验中，原始 GraphTool candidate 的表现很差，主要因为 predicted graph 缺 containment 关系，如 `ON`、`IN_ROOM`、`IN_REGION`。对于 `object_location` QA，如果图里没有这类关系，GraphTool 即使找到对象，也无法给出正确当前位置。

### 4.3 VLM/LLM controls

VLM-only、multi-frame VLM、caption-memory 和 graph-text LLM controls 用于回答同一批 QA。它们的作用是提供非 DSG 或弱 DSG 的对照组。

当前四组 control 在 60 条 QA 上 exact match 都是 0。这说明当前 prompt/answer normalization/QA 难度组合对这些 control 很不友好；不能仅凭这个结果声称 DSG route 已经强，只能说明当前 control pipeline 也需要继续校准。

### 4.4 QA eval、graph eval 与 error attribution

QA eval 评估 prediction 是否与 oracle answer 精确匹配，同时记录 evidence node/edge recall。

Graph eval 比较 predicted graph 与 oracle graph，包括 object precision/recall、relation precision/recall/F1。

Error attribution 将 QA 失败归因到几类：

- `evidence_missing`：预测图缺对象、缺关系或缺状态。
- `graph_construction`：对象存在但构图关系或状态错误。
- `benchmark_or_engine_error`：oracle/engine 侧也无法稳定回答或存在 benchmark 对齐问题。
- `correct`：回答正确。

## 5. 实验数据与指标

### 5.1 原始实验包 readiness

原始 experiment record 显示：

- `readiness_status: ready`
- `real_package_status: ready`
- real package readiness 无 failed checks。
- 研究问题覆盖：`spatial_qa`、`dynamic_memory`、`graph_tool_query`、`interactive_task`。

这表示实验包流程和 artifact 结构通过 gate，但不表示模型效果已经好。

### 5.2 Offline controls

Offline control matrix：

- source kind 数：4。
- source kinds：`caption_memory`、`graph_text`、`multi_frame_vlm`、`vlm`。
- total imported prediction count：240。
- 每组 source 均有 60 条 prediction。

QA exact match：

| Source | Case Count | Exact Match |
| --- | ---: | ---: |
| VLM-only | 60 | 0 |
| Multi-frame VLM | 60 | 0 |
| Caption-memory | 60 | 0 |
| Graph-text LLM | 60 | 0 |
| 原始 predicted GraphTool | 60 | 1 |

原始 predicted GraphTool exact match rate 为 `0.016667`。

### 5.3 原始 predicted graph 质量

原始 graph eval：

| 指标 | 数值 |
| --- | ---: |
| Oracle object count | 176 |
| Predicted object count | 38 |
| Matched object count | 27 |
| Object precision | 0.710526 |
| Object recall | 0.153409 |
| Oracle relation count | 10256 |
| Predicted relation count | 657 |
| Matched relation count | 117 |
| Relation precision | 0.178082 |
| Relation recall | 0.011408 |
| Relation F1 | 0.021442 |

解释：object precision 还可以，说明预测出来的对象很多是对的；但 object recall 极低，说明漏掉了大量 oracle 对象。Relation recall 更低，说明 predicted graph 覆盖不到大多数 oracle 关系。

### 5.4 `candidate_v2` 诊断增强

`candidate_v2` 在不覆盖原始 graph 的前提下，基于已保存 observation 和 AI2-THOR 几何信息，额外补充 containment 关系：

- `IN_REGION`
- `IN_ROOM`
- `ON`

新增 containment edge 数：275。

相关路径：

- v2 graph：`outputs/predicted-dsg/predicted-graph-containment-v2.json`
- v2 prediction：`inputs/candidate/predicted-graph-tool-v2.jsonl`
- v2 dashboard：`inputs/review-v2/index.html`
- v2 readiness：`outputs/diagnostics/candidate-v2-readiness.json`
- v2 experiment record：`outputs/diagnostics/candidate-v2-experiment-record.json`

v2 QA eval：

| 系统 | Case Count | Exact Match | Exact Match Rate |
| --- | ---: | ---: | ---: |
| 原始 predicted GraphTool | 60 | 1 | 0.016667 |
| candidate_v2 GraphTool | 60 | 6 | 0.100000 |

visibility-aligned slice，也就是目标对象都存在于 predicted graph 的 13 条 QA：

| 系统 | Case Count | Exact Match | Exact Match Rate |
| --- | ---: | ---: | ---: |
| 原始 predicted GraphTool | 13 | 0 | 0.000000 |
| candidate_v2 GraphTool | 13 | 5 | 0.384615 |

解释：v2 的提升说明 containment 关系缺失确实是主要失败原因之一。尤其在 visibility-aligned 子集上，对象已存在，补关系后准确率明显提高。

### 5.5 v2 graph eval

v2 graph eval：

| 指标 | 原始 predicted graph | candidate_v2 |
| --- | ---: | ---: |
| Predicted object count | 38 | 38 |
| Matched object count | 27 | 27 |
| Object precision | 0.710526 | 0.710526 |
| Object recall | 0.153409 | 0.153409 |
| Predicted relation count | 657 | 932 |
| Matched relation count | 117 | 294 |
| Relation precision | 0.178082 | 0.315451 |
| Relation recall | 0.011408 | 0.028666 |
| Relation F1 | 0.021442 | 0.052556 |

解释：v2 没有新增对象，所以 object recall 不变。它只补关系，因此 relation precision、recall 和 F1 均提升。但 relation recall 仍只有 `0.028666`，说明关系覆盖仍远低于 oracle。

### 5.6 Error attribution

v2 error attribution 总计 60 条：

| 类别 | 数量 |
| --- | ---: |
| correct | 6 |
| evidence_missing | 29 |
| benchmark_or_engine_error | 24 |
| graph_construction | 1 |

Evidence error category：

| 类别 | 数量 |
| --- | ---: |
| missing_object | 26 |
| missing_relation | 6 |
| missing_state | 20 |
| none | 8 |

按 predicted evidence source 看：

- `missing_predicted_evidence`：44 条 case，43 条错误。
- `ai2thor`：16 条 case，11 条错误。
- `containment_v2`：9 条 case，4 条错误。

解释：即使 v2 补了 containment，仍有 44/60 QA 缺 predicted evidence。这说明最主要瓶颈仍是 observation/detector coverage，不是单个 QA 后处理规则。

### 5.7 Active task

active task 使用 oracle 作为上限 baseline，predicted graph 作为 candidate：

| 指标 | 原始 predicted graph | candidate_v2 |
| --- | ---: | ---: |
| Task success | 1/30 | 5/30 |
| Answer accuracy | 1/30 | 6/30 |
| Answer graph consistency | 1/30 | 5/30 |
| Evidence coverage average | 0.128333 | 0.253333 |

解释：v2 对 active task 也有改善，但仍明显低于 oracle 的 30/30。任务失败仍主要来自 evidence coverage 不足。

## 6. 阶段结论

### 6.1 流程结论

当前小规模真实包的流程是完整的：

- 真实 episode、frame asset、QA、oracle graph 已保存。
- 四组 VLM/LLM/offline control prediction 已导入并评估。
- observation-backed predicted DSG 已生成。
- graph_tool candidate prediction、QA eval、delta、graph eval、error attribution、dashboard、readiness、experiment record 已生成。
- 新增 trace 将图片、QA、prediction、raw response 和错误归因串联起来。

因此，从 artifact 和审计链路角度，本阶段可视为 `process ready`。

### 6.2 效果结论

当前不能宣称结果质量已经达标。主要原因：

- predicted graph 只有 38 个对象，而 oracle graph 有 176 个对象。
- object recall 只有 `0.153409`。
- 60 条 QA 里，只有 13 条的目标对象完全存在于 predicted graph。
- 44/60 QA 仍缺 predicted evidence。
- v2 虽然将 full QA exact match 从 1/60 提升到 6/60，但总体准确率仍只有 0.10。

因此，本阶段结论是：实验链路已经可审计，失败根因已经定位到 evidence coverage 和 relation construction，但 detector/RGB-D predicted DSG 质量还不足以支撑最终正向结论。

### 6.3 方法结论

`candidate_v2` 的提升证明 containment 关系对 GraphTool 很关键。对于 `object_location` 和相对关系 QA，只知道对象存在是不够的；图里必须显式记录 `ON`、`IN_ROOM`、`IN_REGION` 等可查询关系。

但 v2 没有解决 missing object 问题。下一阶段优先级应放在提升 observation coverage、目标对象召回和状态跟踪，而不是继续微调 GraphTool answer formatting。

## 7. 下一阶段 TODO

P0：

- 提高 detector/RGB-D observation coverage，让 predicted graph 覆盖更多 QA 目标对象。
- 对 QA 生成做 observation-aware slice，区分“完整 oracle QA”和“predicted-observable QA”。
- 对每次外部 VLM/LLM 调用保存逐 case request payload、图片引用、raw response、parsed answer 和 parse error。

P1：

- 将 containment relation builder 从诊断侧包转为可验证的标准 predicted DSG 构图模块。
- 针对 `missing_state` 增加状态节点构建和状态时间线校验。
- 重新校准 VLM/control prompt 和 answer normalization，避免四组 control 全部 0/60 时无法解释模型差异。
- 扩大到更多 episode，并保持每轮都输出本格式的中文阶段报告。

## 8. 本阶段报告引用的关键 artifact

- `outputs/experiment-record.json`
- `outputs/real-experiment-readiness.json`
- `outputs/offline-controls/offline-control-matrix.json`
- `outputs/offline-controls/offline-control-result.json`
- `outputs/offline-controls/qa-eval/candidate/predicted_graph_tool/qa-eval.json`
- `inputs/review/graph-eval.json`
- `inputs/review/active-task-delta.json`
- `outputs/diagnostics/diagnostic-report.json`
- `outputs/diagnostics/predicted-graph-tool-v2-qa-eval.json`
- `outputs/diagnostics/predicted-graph-tool-v2-vs-original-delta.json`
- `outputs/diagnostics/visibility-v2-vs-original-delta.json`
- `inputs/review-v2/graph-eval-containment-v2.json`
- `inputs/review-v2/error-attribution-v2.json`
- `inputs/review-v2/active-task-delta-v2.json`
- `inputs/review-v2/dashboard.json`
- `inputs/review-v2/index.html`
- `inputs/traces/trace-readiness.json`
- `inputs/traces/qa-trace.jsonl`
- `inputs/traces/vlm-interactions.jsonl`
- `outputs/diagnostics/candidate-v2-readiness.json`
- `outputs/diagnostics/candidate-v2-experiment-record.json`

## 9. P0-P1 执行补充

本轮执行的是前一节 TODO 中的 P0-P1，而不是重新采集新 simulator episode。重点是把 `candidate_v2` 中证明有效的 containment 诊断逻辑沉淀为正式、可验证的 predicted DSG 构图入口，并把 QA 可观测性与 offline control 校准诊断补齐。

### 9.1 P0：正式 containment inference 构图入口

本轮已在 observation ingestion 和 `scripts/build_predicted_graph.py` 中加入正式参数：

- `--infer-containment`
- `--containment-axis y`

它从 observation sequence 的对象 bbox 与可见区域信息中推断：

- `IN_REGION`
- `IN_ROOM`
- `ON`

这和早先的 `candidate_v2` 侧包不同：`candidate_v2` 是诊断 sidecar，本轮 P0 是标准构图入口的一部分。新产物路径：

- `outputs/predicted-dsg/predicted-graph-containment-p0.json`
- `outputs/predicted-dsg/predicted-graph-containment-p0-report.json`

P0 predicted graph 概况：

| 指标 | 数值 |
| --- | ---: |
| Object count | 38 |
| Edge count | 960 |
| `IN_REGION` edges | 123 |
| `IN_ROOM` edges | 123 |
| `ON` edges | 57 |
| `NEAR` edges | 484 |
| `STATE_CHANGED` edges | 173 |

解释：P0 没有新增 detector 观测对象，所以 object count 仍是 38。提升来自关系补全，尤其是 GraphTool 查询 object location 时需要的 containment 关系。

### 9.2 P0：QA observability split

本轮新增 QA 可观测性分析，区分三类问题：

- `evidence_observable`：所需对象和证据关系都在 predicted graph 中。
- `target_observable_relation_missing`：目标对象存在，但所需关系缺失。
- `target_missing` / `missing_evidence`：目标或证据不在 predicted graph 中。

关键路径：

- `outputs/diagnostics/qa-observability-original.json`
- `outputs/diagnostics/qa-observability-p0.json`
- `inputs/traces/evidence-observable-p0-qa.jsonl`
- `inputs/traces/target-observable-p0-qa.jsonl`
- `inputs/traces/missing-evidence-p0-qa.jsonl`

对比结果：

| Slice | 原始 predicted graph | P0 containment graph |
| --- | ---: | ---: |
| Full QA | 60 | 60 |
| Evidence observable | 1 | 8 |
| Target observable | 14 | 14 |
| Target observable but relation missing | 13 | 6 |
| Target missing | 46 | 46 |
| Missing evidence | 59 | 52 |

解释：P0 将 evidence-observable QA 从 1 条提升到 8 条，说明 containment inference 实际减少了关系缺失。但 target missing 仍是 46 条，说明最大瓶颈仍是 detector/observation coverage。

### 9.3 P0：QA eval、graph eval 与 active task

P0 GraphTool prediction：

- `inputs/candidate/predicted-graph-tool-p0.jsonl`

P0 QA eval：

- `outputs/diagnostics/predicted-graph-tool-p0-qa-eval.json`
- `outputs/diagnostics/predicted-graph-tool-p0-vs-original-delta.json`

QA 结果：

| 系统 | Case Count | Exact Match | Exact Match Rate |
| --- | ---: | ---: | ---: |
| 原始 predicted GraphTool | 60 | 1 | 0.016667 |
| P0 containment GraphTool | 60 | 6 | 0.100000 |

P0 graph eval：

- `inputs/review-p0/graph-eval-containment-p0.json`

| 指标 | 数值 |
| --- | ---: |
| Oracle object count | 176 |
| Predicted object count | 38 |
| Matched object count | 27 |
| Object precision | 0.710526 |
| Object recall | 0.153409 |
| Oracle relation count | 10256 |
| Predicted relation count | 960 |
| Matched relation count | 303 |
| Relation precision | 0.315625 |
| Relation recall | 0.029544 |
| Relation F1 | 0.054030 |

P0 active task：

- `inputs/review-p0/active-candidate-p0-report.json`
- `inputs/review-p0/active-task-delta-p0.json`

| 指标 | P0 |
| --- | ---: |
| Task success | 5/30 |
| Answer accuracy | 6/30 |
| Answer graph consistency | 5/30 |
| Evidence coverage average | 0.253333 |

解释：P0 相比原始图有明确改善，但距离 oracle 上限仍远。active task 失败主要还是 evidence coverage 不足。

### 9.4 P0：error attribution

P0 error attribution 路径：

- `inputs/review-p0/error-attribution-p0.json`

错误归因摘要：

| 类别 | 数量 |
| --- | ---: |
| correct | 6 |
| evidence_missing | 29 |
| benchmark_or_engine_error | 24 |
| graph_construction | 1 |

Evidence error category：

| 类别 | 数量 |
| --- | ---: |
| missing_object | 26 |
| missing_relation | 6 |
| missing_state | 20 |
| none | 8 |

解释：P0 把一部分 missing relation 解决了，但 missing object 和 missing state 仍然多。后续不能只继续加关系规则，需要提高观测覆盖与状态构建。

### 9.5 P1：offline control 校准诊断

本轮新增：

- `outputs/diagnostics/offline-control-calibration-p1.json`

四组 control 的导入是完整的：

| Source | Imported | Missing | Duplicate | Exact Match |
| --- | ---: | ---: | ---: | ---: |
| VLM-only | 60 | 0 | 0 | 0 |
| Multi-frame VLM | 60 | 0 | 0 | 0 |
| Caption-memory | 60 | 0 | 0 | 0 |
| Graph-text LLM | 60 | 0 | 0 | 0 |

答案形态诊断：

| Source | Unknown-like | Relation/location text | Coordinate-like | Other |
| --- | ---: | ---: | ---: | ---: |
| VLM-only | 44 | 11 | 0 | 5 |
| Multi-frame VLM | 35 | 12 | 0 | 13 |
| Caption-memory | 57 | 0 | 0 | 3 |
| Graph-text LLM | 47 | 0 | 9 | 4 |

解释：四组 control 不是因为 import 失败而 0/60，而是答案形态没有与 DSG QA evaluator 对齐。VLM 有部分自然语言位置回答，但没有结构化 evidence；graph-text 输出了坐标类文本，当前 exact-match evaluator 不把它直接等价为 `ON`、`IN_ROOM` 等图关系答案。

### 9.6 P0 readiness 与 final record

本轮新增：

- `outputs/diagnostics/p0-readiness.json`
- `outputs/diagnostics/p0-experiment-record.json`
- `inputs/review-p0/dashboard.json`
- `inputs/review-p0/index.html`

P0 readiness 结论：

- `process_artifact_status: ready`
- `result_quality_status: needs_detector_coverage_work`

也就是说，P0 的流程和 artifact 已经 ready；但结果质量还没有 ready，原因是 predicted graph object recall 仍只有 `0.153409`，且 `target_missing` 仍为 46/60。

## 10. P0-P1 当前结论

本轮 P0-P1 已完成三件关键事：

- 将 containment inference 正式接入 observation-backed predicted DSG 构图入口。
- 建立 QA observability split，能解释哪些 QA 对 predicted graph 来说本来就不可答。
- 建立 offline control 校准诊断，确认 control 低分主要来自答案/证据形态不对齐，而不是导入失败。

当前最重要的结论没有变：实验链路已经完整且可审计，但实验效果仍不理想。下一阶段的首要目标应是提高 detector/RGB-D observation coverage、目标对象召回和状态时间线构建；否则继续微调 GraphTool 或 prompt，只会在少数 already-observable QA 上小幅提升。

## 11. Coverage-v1 诊断实验

### 11.1 实验目标

本轮执行新的 P0-P1 计划，目标是验证：如果显著提高 detector/RGB-D observation coverage，并补齐状态时间线，GraphTool 的 QA 表现是否会明显提升。

本轮同时新增 observation-aware QA slice 和结构化 VLM/LLM control prompt。注意：coverage-v1 使用已保存 AI2-THOR episode metadata 转换成 detector-style observation JSONL，包含 hidden object，因此它是 coverage/state-timeline 诊断实验，不是最终 detector-only 结果。

### 11.2 实验过程

新增和复用的主要 artifact：

- coverage detector JSONL：`inputs/predicted-dsg/detector-rgbd-coverage-v1.jsonl`
- coverage observation sequence：`outputs/predicted-dsg/detector-observations-coverage-v1.json`
- coverage predicted graph：`outputs/predicted-dsg/predicted-graph-coverage-v1.json`
- full QA prediction：`inputs/candidate/predicted-graph-tool-coverage-v1.jsonl`
- evidence-observable QA slice：`inputs/traces/evidence-observable-coverage-v1-qa.jsonl`
- graph eval：`inputs/review-coverage-v1/graph-eval-coverage-v1.json`
- error attribution：`inputs/review-coverage-v1/error-attribution-coverage-v1.json`
- dashboard：`inputs/review-coverage-v1/index.html`
- readiness：`outputs/diagnostics/coverage-v1-readiness.json`
- experiment record：`outputs/diagnostics/coverage-v1-experiment-record.json`
- structured control prompt：`inputs/offline-controls/structured-control-prompt-v1.md`
- structured control schema：`inputs/offline-controls/structured-control-output-schema-v1.json`

coverage detector 构建结果：

| 指标 | 数值 |
| --- | ---: |
| Frame count | 50 |
| Object observation count | 2890 |
| Visible object observation count | 123 |
| Hidden object observation count | 2767 |
| Unique object count | 288 |

解释：这个构建把 episode metadata 中每帧对象转为标准 observation record，并保留 RGB/depth/segmentation 资产路径。它显著提高对象覆盖，但也因为包含 hidden objects，不能当作纯视觉 detector 的真实能力。

### 11.3 Observation-aware QA slice

QA observability 对比：

| 系统 | Case Count | Target Observable | Evidence Observable | Missing Evidence |
| --- | ---: | ---: | ---: | ---: |
| P0 | 60 | 14 | 8 | 52 |
| Coverage-v1 | 60 | 60 | 49 | 11 |

解释：P0 大量 QA 从 predicted graph 角度不可答，主要是目标对象或状态证据不存在。coverage-v1 让 60 条 QA 的目标对象全部进入 predicted graph，并把 evidence-observable QA 提高到 49 条。这说明上一阶段“实验不理想”的核心原因之一确实是观测覆盖不足。

### 11.4 QA eval 与 delta

full QA 结果：

| 系统 | Case Count | Exact Match | Exact Match Rate |
| --- | ---: | ---: | ---: |
| Original predicted GraphTool | 60 | 1 | 0.016667 |
| P0 GraphTool | 60 | 6 | 0.100000 |
| Coverage-v1 GraphTool | 60 | 22 | 0.366667 |

Coverage-v1 相比 P0：

- exact match 增加 16 条。
- exact match rate 增加 0.266667。

Evidence-observable slice 结果：

| 系统 | Case Count | Exact Match | Exact Match Rate |
| --- | ---: | ---: | ---: |
| P0 on coverage-v1 evidence slice | 49 | 6 | 0.122449 |
| Coverage-v1 on evidence slice | 49 | 22 | 0.448980 |

解释：在 observation-aware slice 上，coverage-v1 仍比 P0 多答对 16 条。这说明提升不只是来自“删掉不可答问题”，也来自图中对象和状态证据确实变完整了。

### 11.5 Graph eval

Coverage-v1 graph eval：

| 指标 | 数值 |
| --- | ---: |
| Oracle object count | 176 |
| Predicted object count | 288 |
| Matched object count | 176 |
| Object precision | 0.611111 |
| Object recall | 1.000000 |
| Relation precision | 0.056883 |
| Relation recall | 0.360667 |
| Relation F1 | 0.098268 |
| State accuracy | 0.994318 |

解释：object recall 已经达到 1.0，说明对象缺失不再是 coverage-v1 的主瓶颈。object precision 下降，是因为 metadata coverage 引入了更多 oracle 之外或重复 disambiguation 后的对象节点。relation precision 很低，说明当前构图生成了过多候选关系，下一步需要收紧 `NEAR`、`ON`、`IN_ROOM` 等边的规则和置信度。

### 11.6 Error attribution 与状态时间线

Coverage-v1 error attribution：

| Evidence error category | 数量 |
| --- | ---: |
| none | 49 |
| missing_relation | 11 |
| missing_object | 0 |
| missing_state | 0 |

与 P0 相比：

| 类别 | P0 | Coverage-v1 |
| --- | ---: | ---: |
| missing_object | 26 | 0 |
| missing_state | 20 | 0 |
| missing_relation | 6 | 11 |

解释：状态时间线补齐有效，`missing_state` 从 20 降到 0。对象覆盖也有效，`missing_object` 从 26 降到 0。剩余问题集中到 relation construction：有些 QA 需要的关系边仍缺失，另一些边虽然存在但方向、时刻或 containment 选择不够精确。

### 11.7 Structured VLM/LLM control prompt

本轮新增结构化 prompt 和 schema，要求外部 VLM/LLM control 输出：

- `answer`：与 evaluator 对齐的结构化 JSON。
- `evidence`：图片、caption、memory、detector observation 或 graph-text 证据。
- `observability`：目标是否可见、是否被观测、证据是否充分。
- `error`：当证据不足时输出稳定错误码，而不是编答案。

这一步解决上一轮 control 的主要问题：VLM/LLM 给了自然语言或坐标式回答，但没有结构化 answer/evidence，导致 exact-match evaluator 无法可靠比较。

### 11.8 阶段结论

Coverage-v1 达成了本轮 P0-P1 的主要诊断目标：

- object recall 从 P0 的 0.153409 提升到 1.000000。
- evidence-observable QA 从 8/60 提升到 49/60。
- missing_state 从 20 条降到 0 条。
- full QA exact match 从 P0 的 6/60 提升到 22/60。
- evidence-observable slice exact match 达到 22/49。
- active task success 从 P0 的 5/30 提升到 12/30。

最终结论：实验过程和 artifact 已经完整保存，coverage/state-timeline 的假设被验证；但 coverage-v1 仍不是最终真实 detector-only 结果。下一阶段应把 metadata coverage 中被证明有效的对象覆盖和状态时间线能力迁移回真实 detector/RGB-D pipeline，并重点降低 relation over-generation，提高 relation precision。
