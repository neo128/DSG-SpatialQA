# active QA v2 三组对比

- ready: True
- research_ready: True
- blockers: -

| method | semantic | strict | predictions |
| --- | ---: | ---: | ---: |
| vlm_only | 601/3104 | 601/3104 | 3104 |
| graph_tool_only_dsg | 3104/3104 | 3104/3104 | 3104 |
| vlm_dsg_trusted | 1029/3104 | 1029/3104 | 3104 |

## 结论边界
- 缺少 active QA v2 对齐 VLM-only / VLM+DSG prediction 时，不能形成 superiority claim。
- GraphTool-only 若由 active QA graph records 派生，只能作为图查询消融，不是外部模型结果。
