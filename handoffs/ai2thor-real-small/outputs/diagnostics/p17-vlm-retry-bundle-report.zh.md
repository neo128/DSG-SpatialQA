# P17 VLM-only 失败样本重跑包阶段报告

## 目标

本阶段目标是先提高 VLM-only control 的成功率，再继续优化 DSG。由于当前项目约束不允许默认链路调用外部 VLM/API，本阶段没有发起网络或模型调用，而是把上一轮 VLM-only 失败样本整理成一个可直接交给真实 VLM 生产方的重跑包。

## 原理

上一轮 VLM-only 语义评估结果为 24/60，主要失败原因是模型在可观测目标上仍输出 `target_not_observed` 或无法稳定选择空间支撑物。P17 不向模型暴露 gold answer、gold evidence、语义评估 gold 字段或失败原因，而是：

1. 读取 p14 VLM-only semantic eval；
2. 找出 `semantic_match=false` 的 case；
3. 从 p7 增强 VLM request bundle 中只保留这些失败 case；
4. 保留非 gold 的 target crop、support candidates、answer options 和结构化 JSON 输出约束；
5. 删除所有 evaluator-only / gold / prior prediction 字段；
6. 输出一个只需重跑失败样本的 VLM retry bundle。

这样可以让外部 VLM 重跑集中在 36 个失败样本上，成功样本 24 个可以保留不变。下一轮若返回 36 条新 prediction JSONL，可与原成功 prediction 合并后重新计算 VLM-only 分数。

## 产物

- Retry bundle: `handoffs/ai2thor-real-small/offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p17-retry-failures.json`
- Source enhanced bundle: `handoffs/ai2thor-real-small/offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p7-independent-support-crop-options.json`
- Source eval report: `handoffs/ai2thor-real-small/outputs/diagnostics/vlm-semantic-eval-p14-affordance-room-specific-support.json`
- Pending empty retry return: `handoffs/ai2thor-real-small/inputs/offline-controls/reruns/observation-aware-p4-target60/vlm-p17-retry-pending-empty.jsonl`
- Pending merged predictions: `handoffs/ai2thor-real-small/inputs/offline-controls/reruns/observation-aware-p4-target60/vlm-p17-merged-pending.jsonl`
- Pending merge report: `handoffs/ai2thor-real-small/outputs/diagnostics/vlm-p17-retry-merge-pending-report.json`
- Pending semantic eval: `handoffs/ai2thor-real-small/outputs/diagnostics/vlm-semantic-eval-p17-merged-pending.json`
- Retry input gap report: `handoffs/ai2thor-real-small/outputs/diagnostics/vlm-p18-retry-input-gap-report.json`

## 数据

- 原 VLM-only p14：24/60，semantic match rate = 0.40；
- p17 retry case：36/60；
- 保留成功 case：24/60；
- retry case 中 target crop 缺失：19；
- retry case 中 support candidates 缺失：5；
- retry case 中 answer options 缺失：0；
- answer option 数量范围：1 到 6；
- forbidden key 审计：未发现 `gold`、`gold_answer`、`gold_evidence`、`semantic_match`、`failure_reason`、`prediction`、`evaluator_only` 等 JSON key。

## 合并验收入口

本阶段新增了 retry prediction 合并入口。外部 VLM 返回 36 条 retry prediction JSONL 后，可以将其与原始 60 条 p14 prediction 合并：语义评估已成功的 24 条保留原预测，失败的 36 条用 retry prediction 覆盖。

当前用 `vlm-p17-retry-pending-empty.jsonl` 模拟“尚未收到外部 retry 结果”的状态，合并报告为 ready=false：

- expected case：60；
- kept original success：24；
- retry expected：36；
- retry received：0；
- missing retry：36；
- merged prediction：60；
- pending semantic eval：24/60，semantic match rate = 0.40。

这说明空返回不会虚假提高 VLM-only 分数。只有收到完整 36 条真实 retry prediction 后，merge report 才能 ready=true，并触发新的 VLM-only semantic eval。

## P18 输入缺口

P18 对 P17 retry bundle 做了输入完整性诊断，目标是先提高 VLM-only 的可视输入质量，再跑真实 VLM：

- retry case：36；
- answer options：36/36；
- frames：36/36；
- primary frame：36/36；
- support candidates：31/36，缺 5；
- target crop：17/36，缺 19；
- next collection targets：23；
- forbidden key 审计：未发现 `gold`、`gold_answer`、`gold_evidence`、`semantic_match`、`failure_reason`、`prediction`、`evaluator_only` 等 JSON key。

结论是：P17 已经具备结构化选项和多帧图像输入，但仍有 19 条缺 target crop、5 条缺 support candidates。下一轮应优先让 detector/VLM preprocessing 补齐这些 target crop 与 support candidates，再交给真实 VLM 重跑；否则 VLM-only 很可能继续把部分可见目标判成 `target_not_observed`。

## 结论

P17 不能直接证明 VLM-only 已经提升，因为尚未调用真实 VLM 生成新预测。它完成的是下一次真实 VLM 重跑前最关键的输入优化：把失败样本收窄到 36 条，并保证每条都有结构化 answer option 约束，同时不泄露 gold 信息。

下一步应让外部 VLM 对 P17 retry bundle 生成真实 prediction JSONL，然后用新增合并入口把新失败样本预测与 p14 的 24 个成功样本合并，重新评估 VLM-only。如果 VLM-only 显著上升，再用同一 QA slice 对 DSG predicted graph 进行公平比较。

## P19 detector bbox crop 补强

P19 继续沿着“先提高 VLM-only 输入质量”的方向推进。本轮没有调用外部 VLM/API，也没有生成新的预测分数；只使用已经保存在本地的 detector/RGB-D JSONL，为 P17 的 36 条失败重跑 case 补 target crop。

使用的本地 detector 输入：

- `handoffs/ai2thor-real-small/inputs/predicted-dsg/detector-rgbd-coverage-ai2thor-visible-p2.jsonl`
- `handoffs/ai2thor-real-small/inputs/predicted-dsg/detector-rgbd-coverage-ai2thor-visible-p3-offset.jsonl`
- `handoffs/ai2thor-real-small/inputs/predicted-dsg/detector-rgbd-independent-p4-target60.jsonl`

新增产物：

- P19 retry bundle: `handoffs/ai2thor-real-small/offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p19-detector-bbox-retry-failures.json`
- P19 crop root: `handoffs/ai2thor-real-small/inputs/frame-assets/vlm-target-crops-p19-detector-bbox/`
- P19 input gap report: `handoffs/ai2thor-real-small/outputs/diagnostics/vlm-p19-retry-input-gap-report.json`

P19 关键数据：

- retry case：36；
- target crop：35/36，P18 为 17/36；
- target crop 缺失：1，P18 为 19；
- support candidates：31/36，仍缺 5；
- answer options：36/36；
- frames：36/36；
- primary frame：36/36；
- 写出的 target crop 文件数：35；
- request bundle digest：`39f20eb109e5f46a83b7ddaadcf310a5d59c0618cf0444083b3163f0cd2b7dfd`。

剩余缺口：

- 1 条缺 target crop：`cd_01_62_00_80_01_19`；
- 5 条缺 support candidates：`armchair_04_38_00_00_06_02`、`baseballbat_02_84_00_29_01_82`、`garbagecan_00_05_00_00_03_88`、`cabinet_00_63_02_02_01_45`、`cabinet_01_70_02_02_01_45`。

解释：P19 明显提高了 VLM-only 重跑包的可视输入完整性，尤其是将 target crop 缺口从 19 条降到 1 条。这有助于减少 VLM 把可见目标误判为 `target_not_observed` 的概率。但在收到真实 VLM retry prediction 之前，VLM-only 成功率仍不能声称提升；当前只能说 VLM-only 的重跑输入质量已经显著改善。

## P20 support gap 复判

P20 没有新增外部模型调用，而是修正 retry input gap 的判定：如果一个 case 的 answer options 只有 `IN_ROOM/room`，那么缺少 `support_candidates` 不再视为采集缺口，因为这类问题没有可供选择的非房间支撑物；相反，模型应基于目标 crop 和主帧判断目标是否位于房间级位置。

新增产物：

- P20 retry bundle: `handoffs/ai2thor-real-small/offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p20-ready-except-cd-crop.json`
- P20 input gap report: `handoffs/ai2thor-real-small/outputs/diagnostics/vlm-p20-retry-input-gap-report.json`

P20 关键数据：

- retry case：36；
- answer options：36/36；
- frames：36/36；
- primary frame：36/36；
- target crop：35/36；
- support candidates required：31；
- support candidates present：31；
- support candidates not applicable：5；
- actionable support gap：0；
- remaining actionable gap：1 条 target crop。

剩余唯一缺口：

- `ai2thor-real-small-episode-003:FloorPlan301:0010:object_location:cd_01_62_00_80_01_19:observation_aware:100029`

解释：P20 后，VLM-only retry 包已经不再有 support candidate 层面的待补缺口。唯一剩余问题是 `cd` 目标在所有本地 detector 记录里可见但缺 2D bbox，因此不能安全生成 target crop。下一步如果要彻底消除该缺口，需要重新采集或运行 detector，让 `cd_01_62_00_80_01_19` 产出显式 `bbox_2d_xyxy`；不应使用整帧冒充 target crop。

## P21 ready35 分流重跑包

P21 继续执行“先提高 VLM-only，再优化 DSG”的顺序。本轮没有调用外部 VLM/API，也没有伪造预测分数；它做的是把 P20 中已经输入完整的 35 条失败样本拆出来，形成一个可立即交给外部 VLM 的 retry subset，同时把唯一缺 target crop 的 `cd` case 保留为显式 blocker。

新增产物：

- P21 ready35 retry bundle: `handoffs/ai2thor-real-small/offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p21-ready35-crop-complete.json`
- P21 input gap report: `handoffs/ai2thor-real-small/outputs/diagnostics/vlm-p21-ready35-input-gap-report.json`
- P21 `cd` crop blocker report: `handoffs/ai2thor-real-small/outputs/diagnostics/vlm-p21-cd-crop-blocker-report.json`

P21 ready35 关键数据：

- retry subset case：35；
- answer options：35/35；
- frames：35/35；
- primary frame：35/35；
- target crop：35/35；
- support candidates required：30；
- support candidates present：30；
- support candidates not applicable：5；
- missing target crop：0；
- missing support candidates：0；
- input gap report `ready=true`；
- request bundle digest：`7ae2d1555c1b859baba8da537b255eaecbbe8c81fc7f1ab594d8eb004d7bd8fc`；
- input gap report digest：`c6b43599324a75f954a08899f549b26e50465739650a66b5394b9676134316fd`。

P21 对 `cd` 缺口做了额外排查：

- 检查了本地 detector/coverage JSONL 中的 36 条 `cd_01_62_00_80_01_19` detection record；
- 其中有 `bbox_2d_xyxy` 的记录数为 0；
- episode metadata 只有 3D bbox，没有可直接裁剪的 2D bbox；
- segmentation residual 匹配后剩余的是整幅背景/大区域 segment，不能安全当作 `cd` mask；
- 因此不使用整帧或背景 residual 冒充 target crop。

外部 VLM 可先对 35 条 ready subset 运行：

```bash
python external_tools/run_vlm_controls.py \
  --request-bundle handoffs/ai2thor-real-small/offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p21-ready35-crop-complete.json \
  --output handoffs/ai2thor-real-small/inputs/offline-controls/reruns/observation-aware-p4-target60/vlm-p21-ready35.jsonl \
  --trace-output handoffs/ai2thor-real-small/inputs/traces/reruns/vlm-p21-ready35-trace.jsonl \
  --source-kind vlm \
  --api-key-env DSG_SPATIALQA_DASHSCOPE_API_KEY \
  --allow-network
```

解释：P21 让 VLM-only 可以先跑 35 个输入完整的失败样本，不再被 1 个缺 bbox 的 case 卡住。但这仍不是最终 VLM-only 分数提升证据。只有收到真实 VLM prediction JSONL，并与原来 p14 的 24 条成功样本合并后，才能重新计算 VLM-only semantic match。若只返回 35 条，merge report 会明确保留 1 条 missing retry blocker；它可以作为诊断分数，但不能作为最终完整 36-case retry claim。

### P21 pending merge 验证

为避免 ready35 输入包被误读成“VLM-only 已提升”，本轮还生成了空返回的 pending merge 诊断：

- 空 retry prediction：`handoffs/ai2thor-real-small/inputs/offline-controls/reruns/observation-aware-p4-target60/vlm-p21-ready35-pending-empty.jsonl`
- pending merged prediction：`handoffs/ai2thor-real-small/inputs/offline-controls/reruns/observation-aware-p4-target60/vlm-p21-merged-pending.jsonl`
- pending merge report：`handoffs/ai2thor-real-small/outputs/diagnostics/vlm-p21-ready35-merge-pending-report.json`
- pending semantic eval：`handoffs/ai2thor-real-small/outputs/diagnostics/vlm-semantic-eval-p21-merged-pending.json`

pending merge 结果：

- merge report `ready=false`；
- expected case：60；
- original prediction：60；
- retry expected：36；
- retry received：0；
- missing retry：36；
- merged prediction：60；
- kept original success：24。

pending semantic eval 结果仍为：

- semantic match：24/60；
- semantic match rate：0.40；
- strict exact match：0/60。

解释：空 retry 返回不会让 VLM-only 分数上涨。P21 当前只证明 35 条 retry 输入已经可执行；真正的 VLM-only 成功率提升必须等待真实 `vlm-p21-ready35.jsonl` 返回后重新 merge 和 eval。

### P21 scoped merge gate 修正

本轮补了一处和 P21 分流策略直接相关的 merge gate：旧的 retry merge 默认把语义评估中的全部失败样本作为 retry 期望，因此即使只收到 P21 ready35 的 35 条真实返回，也会继续把被显式排除的 `cd` crop blocker 记为 missing retry。现在 `scripts/eval_vlm_calibration.py` 支持通过 `--merge-retry-request-bundle` 使用实际 request bundle 的 `case_inputs` 作为 retry scope，并且当 bundle 的 enrichment 里残留旧 `retry_case_ids` 时，以实际 `case_inputs` 为准。

新增产物：

- scoped pending merged prediction: `handoffs/ai2thor-real-small/inputs/offline-controls/reruns/observation-aware-p4-target60/vlm-p21-ready35-scope-merged-pending.jsonl`
- scoped pending merge report: `handoffs/ai2thor-real-small/outputs/diagnostics/vlm-p21-ready35-scope-merge-pending-report.json`
- scoped pending semantic eval: `handoffs/ai2thor-real-small/outputs/diagnostics/vlm-semantic-eval-p21-ready35-scope-merged-pending.json`

scoped pending merge 结果：

- retry scope kind: `explicit_retry_case_ids`；
- expected case：60；
- retry expected：35；
- retry received：0；
- missing retry：35；
- out-of-scope failed case：1；
- kept original success：24；
- merged prediction：60；
- merge report digest：`cdb83080f747f82154aba1ce3fb9c15bf7343b11460179ee2a1cfdd1c96bccd6`。

scoped pending semantic eval 仍为：

- semantic match：24/60；
- semantic match rate：0.40；
- strict exact match：0/60；
- semantic eval digest：`368a4f710bac1aef9a23df038e9282b9028f2a9d667b57cabe1e1fb0fcd40d9d`。

解释：这次修正没有提高 VLM-only 分数，但消除了后续接收 35 条真实 retry prediction 时的误拦截。当前 VLM-only 提升的唯一剩余输入是 `vlm-p21-ready35.jsonl`；在它返回前，不能声称 VLM-only 已提升。
