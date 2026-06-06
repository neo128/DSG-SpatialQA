# P51 active QA v2 逐例归因

- case_count: 576
- adjudicated wins: 120
- adjudicated failures: 206
- tie correct: 230
- adjudicated losses: 20

## 归因分布

| attribution | count |
| --- | ---: |
| accepted_vlm_but_wrong | 175 |
| adjudicator_regression | 20 |
| adjudicator_rejected_both | 27 |
| adjudicator_uncertain | 4 |
| both_correct | 230 |
| dsg_location_correction | 25 |
| dsg_situated_evidence_correction | 21 |
| dsg_support_relation_correction | 38 |
| dsg_temporal_memory_correction | 36 |

## 结论边界

- 本报告使用 evaluator gold answer 判断 win/failure，因此只能作为离线诊断。
- 不得把本报告内容放入外部 VLM request bundle。
