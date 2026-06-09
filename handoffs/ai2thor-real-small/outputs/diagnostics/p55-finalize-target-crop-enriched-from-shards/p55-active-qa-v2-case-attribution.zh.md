# P51 active QA v2 逐例归因

- case_count: 3104
- adjudicated wins: 428
- adjudicated failures: 2075
- tie correct: 601
- adjudicated losses: 0

## 归因分布

| attribution | count |
| --- | ---: |
| accepted_vlm_but_wrong | 1794 |
| adjudicator_other_win | 51 |
| adjudicator_rejected_both | 12 |
| both_correct | 601 |
| dsg_evidence_correction | 281 |
| dsg_location_correction | 8 |
| dsg_situated_evidence_correction | 48 |
| dsg_support_relation_correction | 17 |
| dsg_temporal_memory_correction | 23 |
| trusted_gate_missed_correct_graph | 269 |

## 结论边界

- 本报告使用 evaluator gold answer 判断 win/failure，因此只能作为离线诊断。
- 不得把本报告内容放入外部 VLM request bundle。
