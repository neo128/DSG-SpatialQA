# DSG-vs-Control 研究结论报告

结论：当前实验不支持 DSG 优于 VLM / 视频记忆

## 判定依据

- candidate exact match: 1/60 = 0.016667
- required controls passed: 0/4
- minimum exact-match delta: 0.016667
- minimum sign-test p-value: 0.5
- predicted graph object recall: 0.159091
- predicted graph unlocated objects: 0
- evaluation scope: full_oracle
- evidence-observable QA: 3/60

## 主要原因

- candidate_exact_match_rate_below_floor: Candidate exact-match rate is below the practical superiority floor.
- graph_object_recall_below_floor: Predicted DSG object recall is below the evidence floor.
- no_control_passed_superiority: No required control comparison passed all superiority gates.

## 解释边界

该报告只对输入 artifact 所代表的当前实验包负责；它不会把小规模结果推广成通用结论。
