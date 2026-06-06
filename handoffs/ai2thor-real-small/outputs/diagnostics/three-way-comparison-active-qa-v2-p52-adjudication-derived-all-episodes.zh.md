# active QA v2 三组对比

- ready: True
- research_ready: True
- blockers: -

| method | semantic | strict | predictions |
| --- | ---: | ---: | ---: |
| vlm_only | 141/576 | 141/576 | 576 |
| graph_tool_only_dsg | 576/576 | 576/576 | 576 |
| vlm_dsg_trusted | 433/576 | 433/576 | 576 |

## 结论边界
- 缺少 active QA v2 对齐 VLM-only / VLM+DSG prediction 时，不能形成 superiority claim。
- GraphTool-only 若由 active QA graph records 派生，只能作为图查询消融，不是外部模型结果。
