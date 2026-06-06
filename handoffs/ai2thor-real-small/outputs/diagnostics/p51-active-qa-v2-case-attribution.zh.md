# P51 active QA v2 逐例归因

- case_count: 576
- adjudicated wins: 142
- adjudicated failures: 293
- tie correct: 141
- adjudicated losses: 0

## 归因分布

| attribution | count |
| --- | ---: |
| accepted_vlm_but_wrong | 257 |
| adjudicator_rejected_both | 31 |
| adjudicator_uncertain | 5 |
| both_correct | 141 |
| dsg_location_correction | 37 |
| dsg_situated_evidence_correction | 31 |
| dsg_support_relation_correction | 38 |
| dsg_temporal_memory_correction | 36 |

## 结论边界

- 本报告使用 evaluator gold answer 判断 win/failure，因此只能作为离线诊断。
- 不得把本报告内容放入外部 VLM request bundle。
