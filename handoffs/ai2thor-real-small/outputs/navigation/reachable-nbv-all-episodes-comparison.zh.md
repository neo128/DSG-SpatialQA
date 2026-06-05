# reachable NBV 多 episode 协议对比

## 总结
- episode_count: 5
- formal_protocol_ready_episode_count: 5
- all_episodes_formal_protocol_ready: True

## Episode 表

| episode | scene | ready | same_frame fixed→NBV | evidence fixed→NBV | missing_support fixed→NBV | missing_relation fixed→NBV | GraphTool semantic fixed→NBV | failed_checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| episode001 | FloorPlan1 | True | 0.083333→0.75 | 2→12 | 1→0 | 11→3 | 3→9 | - |
| episode002 | FloorPlan201 | True | 0.083333→0.416667 | 2→10 | 1→0 | 11→7 | 2→5 | - |
| episode003 | FloorPlan301 | True | 0.083333→0.5 | 2→12 | 1→0 | 11→6 | 0→6 | - |
| episode004 | FloorPlan401 | True | 0.25→0.5 | 6→11 | 4→1 | 9→6 | 1→5 | - |
| episode005 | FloorPlan2 | True | 0.166667→0.583333 | 4→12 | 3→0 | 10→5 | 1→6 | - |

## 解释边界
- ready=false 的 episode 不能作为正式多 episode 探索结论。
- coverage diagnostic 仍只作为上限诊断，不作为 predicted DSG evidence。
