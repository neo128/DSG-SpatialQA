# active QA v2 三组对比

- ready: False
- research_ready: False
- blockers: directional_not_significant, missing_active_vlm_dsg_trusted_predictions, missing_active_vlm_only_predictions, paired_wins_not_above_losses, vlm_dsg_not_above_vlm_only

| method | semantic | strict | predictions |
| --- | ---: | ---: | ---: |
| vlm_only | 0/488 | 0/488 | 0 |
| graph_tool_only_dsg | 488/488 | 488/488 | 488 |
| vlm_dsg_trusted | 0/488 | 0/488 | 0 |

## 结论边界
- 缺少 active QA v2 对齐 VLM-only / VLM+DSG prediction 时，不能形成 superiority claim。
- GraphTool-only 若由 active QA graph records 派生，只能作为图查询消融，不是外部模型结果。
