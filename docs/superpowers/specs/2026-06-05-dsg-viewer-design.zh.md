# DSG 查看器设计

日期：2026-06-05

## 目标

为 DSG-SpatialQA 项目构建一个本地 DSG 检查工作台。这个工具需要在同一个界面里支持三类工作流：

- 调试 predicted DSG 质量，包括缺失证据、噪声关系、对象定位问题，以及 evidence readiness 失败项。
- 演示 DSG 结构和 QA 证据路径，让别人能在浏览器里直观看懂图结构。
- 分析实验结果，把 predicted/oracle graph 差异、QA 错误、graph metrics 和 evidence reports 联动起来。

第一版是只读的本地 Web app。它不修改 graph artifact，不调用外部服务，也不运行 detector 或 simulator 集成。

## 选定方案

采用 Inspector Workbench 布局：

- 左侧栏：数据源状态、模式 tab、搜索和筛选。
- 中间区域：指标卡、工具栏和交互式 graph canvas。
- 右侧栏：当前选中节点/边详情、状态历史、oracle delta、关联 QA cases 和诊断信息。

这个布局信息密度足够支撑调试和分析；同时可以通过隐藏高级筛选、突出证据路径来支持更干净的演示模式。

## 架构

新增本地 server 入口：

```bash
python scripts/serve_dsg_viewer.py --workspace handoffs/ai2thor-real-small
```

server 必须是本地、确定性的：

- 默认绑定到 localhost。
- 只读取显式 `--workspace` 路径下的文件，或显式传入的 artifact 路径。
- 不进行网络调用。
- 不读取当前时间，不生成随机输出。
- 拒绝访问允许的 project/workspace root 之外的路径。

后端职责：

- 加载本地 JSON/JSONL artifacts。
- 在已有 validator 可用时校验已知 schema version。
- 将 artifacts 归一化为一个 viewer payload。
- 提供静态 viewer 页面和 payload endpoints。

前端职责：

- 渲染 graph canvas、筛选器、指标卡和详情面板。
- 保持 graph selection、QA selection 和 report diagnostics 联动。
- 基于同一个 payload 提供 Debug、Demo 和 Analysis 三种视图。

## 数据输入

第一版支持以下 artifact 类型：

- Predicted graph JSON，例如 `outputs/predicted-dsg/predicted-graph.json`。
- Oracle graph JSON，可选；用于 delta 对比。
- QA dataset JSONL。
- QA eval report JSON。
- Graph eval report JSON。
- Predicted DSG evidence report JSON。

server 应同时支持 workspace preset 和显式文件参数。workspace preset 应能在 `handoffs/ai2thor-real-small` 下发现当前项目常用 artifact 路径。

## Viewer Payload

后端把 artifacts 归一化为一个紧凑 payload，包含这些部分：

- `graph`：nodes、edges、object states、state histories、agent history、graph summary。
- `oracle`：可选 oracle graph summary；如果提供 graph eval，则包含 predicted-to-oracle matches。
- `qa`：cases、predictions/eval rows、success/failure status、target object ids，以及可用时的 missing evidence。
- `metrics`：graph counts、relation counts、object recall、relation precision/recall/F1、evidence readiness。
- `diagnostics`：evidence gate checks、failed checks、source counts 和 report digests。

payload 需要保留原始 artifact 路径和 digest，这样用户能从界面里的任意显示项追溯到已保存文件。

## 核心交互

Graph 交互：

- 按类型显示节点：object、room、region、state、agent。
- 按 relation 显示边。
- 支持按 node type、label、relation、step range 和 error category 筛选。
- 搜索 node id、label、relation name 和 QA case id。
- 点击节点或边后，右侧详情栏展示对应信息。

QA 交互：

- 从关联 QA 列表选择一个 QA case。
- 高亮 target object nodes 和 evidence edges。
- 在可用时显示 missing graph evidence。
- 显示 prediction status、exact match status、semantic match status 和 failure reason。

对比交互：

- 当 oracle/graph eval 数据存在时，标记 matched、missing 和 extra objects/relations。
- 对当前选中项显示 oracle match 和 delta 详情。
- 中间工具栏提供 Show Delta 开关。

模式 tabs：

- Debug：突出筛选器和诊断信息。
- Demo：突出 graph canvas 和人类可读解释。
- Analysis：突出指标、QA table 和 predicted/oracle deltas。

## 错误处理

viewer 应明确展示 artifact 问题，而不是静默失败：

- 缺少可选文件：显示禁用面板，并展示缺失路径。
- schema 无效：显示 validator error，同时保持其他有效面板可用。
- 路径超出 workspace：拒绝请求。
- JSON/JSONL 格式错误：显示文件路径和解析错误。
- 缺少联动字段：加载 artifact，但将受影响的 join 标记为不可用。

## 测试

生产代码使用 test-first 实现。

建议测试：

- path allowlist 会拒绝 workspace 外的文件。
- workspace preset 能解析预期的 `ai2thor-real-small` artifact 路径。
- graph JSON 归一化能产生稳定的 node/edge/state 数量。
- QA/eval 联动能把 object-location cases 关联到 target object ids。
- evidence report 归一化能暴露 readiness 和 failed checks。
- 缺少可选 artifacts 不会阻塞有效 artifacts 加载。
- CLI/server 参数解析会拒绝含糊或无效输入。

完整验证仍然使用：

```bash
python scripts/verify.py
```

## 第一版非目标

- 不编辑 graph。
- 不持久化人工标注。
- 不做多用户或远程部署。
- 不调用外部 AI、detector、simulator 或服务。
- 不做大型服务端 graph layout engine。
- 不替代 graph eval 或 QA eval；viewer 只展示并联动已有 artifacts。

## 实现约束

第一版应保持轻依赖：使用 Python 标准库 HTTP serving，加上提交到项目里的静态 HTML/JavaScript viewer。除非有 failing test 或明确的浏览器兼容性问题证明标准库/静态方案无法满足第一版需求，否则不要添加新的 package dependency。
