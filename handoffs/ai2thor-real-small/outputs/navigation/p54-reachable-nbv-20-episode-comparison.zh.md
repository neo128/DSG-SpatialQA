# reachable NBV 多 episode 协议对比

## 总结
- episode_count: 20
- formal_protocol_ready_episode_count: 5
- all_episodes_formal_protocol_ready: False

## Episode 表

| episode | scene | ready | same_frame fixed→NBV | evidence fixed→NBV | missing_support fixed→NBV | missing_relation fixed→NBV | GraphTool semantic fixed→NBV | failed_checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| episode001 | FloorPlan1 | True | 0.083333→0.75 | 2→12 | 1→0 | 11→3 | 3→9 | - |
| episode002 | FloorPlan201 | True | 0.083333→0.416667 | 2→10 | 1→0 | 11→7 | 2→5 | - |
| episode003 | FloorPlan301 | True | 0.083333→0.5 | 2→12 | 1→0 | 11→6 | 0→6 | - |
| episode004 | FloorPlan401 | True | 0.25→0.5 | 6→11 | 4→1 | 9→6 | 1→5 | - |
| episode005 | FloorPlan2 | True | 0.166667→0.583333 | 4→12 | 3→0 | 10→5 | 1→6 | - |
| episode006 | FloorPlan202 | False | None→None | None→None | None→None | None→None | None→None | missing_trajectory,missing_decision_trace,missing_fixed_audit,missing_nbv_audit |
| episode007 | FloorPlan302 | False | None→None | None→None | None→None | None→None | None→None | missing_trajectory,missing_decision_trace,missing_fixed_audit,missing_nbv_audit |
| episode008 | FloorPlan402 | False | None→None | None→None | None→None | None→None | None→None | missing_trajectory,missing_decision_trace,missing_fixed_audit,missing_nbv_audit |
| episode009 | FloorPlan3 | False | None→None | None→None | None→None | None→None | None→None | missing_trajectory,missing_decision_trace,missing_fixed_audit,missing_nbv_audit |
| episode010 | FloorPlan203 | False | None→None | None→None | None→None | None→None | None→None | missing_trajectory,missing_decision_trace,missing_fixed_audit,missing_nbv_audit |
| episode011 | FloorPlan303 | False | None→None | None→None | None→None | None→None | None→None | missing_trajectory,missing_decision_trace,missing_fixed_audit,missing_nbv_audit |
| episode012 | FloorPlan403 | False | None→None | None→None | None→None | None→None | None→None | missing_trajectory,missing_decision_trace,missing_fixed_audit,missing_nbv_audit |
| episode013 | FloorPlan4 | False | None→None | None→None | None→None | None→None | None→None | missing_trajectory,missing_decision_trace,missing_fixed_audit,missing_nbv_audit |
| episode014 | FloorPlan204 | False | None→None | None→None | None→None | None→None | None→None | missing_trajectory,missing_decision_trace,missing_fixed_audit,missing_nbv_audit |
| episode015 | FloorPlan304 | False | None→None | None→None | None→None | None→None | None→None | missing_trajectory,missing_decision_trace,missing_fixed_audit,missing_nbv_audit |
| episode016 | FloorPlan404 | False | None→None | None→None | None→None | None→None | None→None | missing_trajectory,missing_decision_trace,missing_fixed_audit,missing_nbv_audit |
| episode017 | FloorPlan5 | False | None→None | None→None | None→None | None→None | None→None | missing_trajectory,missing_decision_trace,missing_fixed_audit,missing_nbv_audit |
| episode018 | FloorPlan205 | False | None→None | None→None | None→None | None→None | None→None | missing_trajectory,missing_decision_trace,missing_fixed_audit,missing_nbv_audit |
| episode019 | FloorPlan305 | False | None→None | None→None | None→None | None→None | None→None | missing_trajectory,missing_decision_trace,missing_fixed_audit,missing_nbv_audit |
| episode020 | FloorPlan405 | False | None→None | None→None | None→None | None→None | None→None | missing_trajectory,missing_decision_trace,missing_fixed_audit,missing_nbv_audit |

## 解释边界
- ready=false 的 episode 不能作为正式多 episode 探索结论。
- coverage diagnostic 仍只作为上限诊断，不作为 predicted DSG evidence。
