# Episode001 三轨迹协议对比

## 结论
- fixed trajectory 覆盖不足: True
- diagnostic 补采提升 evidence coverage: True
- reachable NBV navigation validated: True
- reachable NBV 减少 missing support: False
- reachable NBV 减少 missing relation: True
- reachable NBV 提升 GraphTool semantic match: True
- reachable NBV 可作为正式探索协议: False

## 边界
- 若 runtime_kind=fake_controller，则该结果只验证机制，不是实时 AI2-THOR 真实 rollout 结论。
