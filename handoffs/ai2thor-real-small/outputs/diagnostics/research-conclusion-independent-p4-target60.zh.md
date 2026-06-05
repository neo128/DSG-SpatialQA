# DSG-vs-Control 研究结论报告

结论：实验包未 ready，不能形成优越性结论

## 判定依据

- candidate exact match: 0/60 = 0.0
- required controls passed: 0/2
- minimum exact-match delta: 0.0
- minimum sign-test p-value: 1.0
- predicted graph object recall: 0.164773
- predicted graph unlocated objects: 39
- evaluation scope: observation_aware
- evidence-observable QA: 60/60
- semantic location result: DSG 0/60, VLM-only 6/60, multi-frame VLM 4/60
- semantic location delta: DSG-VLM = -0.1, DSG-multi-frame = -0.066667

## 主要原因

- offline_controls_not_ready: offline_controls_ready
- predicted DSG graph 缺少可靠支撑面/current-location 关系，GraphTool 无法回答
  object-location QA。

## 解释边界

该报告只对输入 artifact 所代表的当前实验包负责；它不会把小规模结果推广成通用结论。
formal readiness 仍为 `inconclusive_not_ready`，但同切片 semantic location
比较已经支持阶段性负向结论：当前 independent detector/RGB-D DSG 没有优于
VLM-only 或视频记忆，且在该语义口径下低于两者。
