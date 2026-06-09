# reachable NBV 多 episode 协议对比

## 总结
- episode_count: 20
- formal_protocol_ready_episode_count: 20
- all_episodes_formal_protocol_ready: True

## Episode 表

| episode | scene | ready | same_frame fixed→NBV | evidence fixed→NBV | missing_support fixed→NBV | missing_relation fixed→NBV | GraphTool semantic fixed→NBV | failed_checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| episode001 | FloorPlan1 | True | 0.083333→0.75 | 2→12 | 1→0 | 11→3 | 3→9 | - |
| episode002 | FloorPlan201 | True | 0.083333→0.416667 | 2→10 | 1→0 | 11→7 | 2→5 | - |
| episode003 | FloorPlan301 | True | 0.083333→0.5 | 2→12 | 1→0 | 11→6 | 0→6 | - |
| episode004 | FloorPlan401 | True | 0.25→0.5 | 6→11 | 4→1 | 9→6 | 1→5 | - |
| episode005 | FloorPlan2 | True | 0.166667→0.583333 | 4→12 | 3→0 | 10→5 | 1→6 | - |
| episode006 | FloorPlan202 | True | 0.026316→1.0 | 2→38 | 8→0 | 37→0 | 2→38 | - |
| episode007 | FloorPlan302 | True | 0.026316→1.0 | 2→38 | 8→0 | 37→0 | 4→38 | - |
| episode008 | FloorPlan402 | True | 0.028571→1.0 | 2→35 | 6→0 | 34→0 | 1→35 | - |
| episode009 | FloorPlan3 | True | 0.027027→1.0 | 2→37 | 5→0 | 36→0 | 5→37 | - |
| episode010 | FloorPlan203 | True | 0.028571→1.0 | 2→35 | 16→0 | 34→0 | 0→35 | - |
| episode011 | FloorPlan303 | True | 0.025641→1.0 | 2→39 | 9→0 | 38→0 | 1→39 | - |
| episode012 | FloorPlan403 | True | 0.027027→1.0 | 2→37 | 6→0 | 36→0 | 5→37 | - |
| episode013 | FloorPlan4 | True | 0.028571→1.0 | 2→35 | 7→0 | 34→0 | 7→35 | - |
| episode014 | FloorPlan204 | True | 0.025641→1.0 | 2→39 | 6→0 | 38→0 | 2→39 | - |
| episode015 | FloorPlan304 | True | 0.026316→1.0 | 2→38 | 9→0 | 37→0 | 0→38 | - |
| episode016 | FloorPlan404 | True | 0.027778→1.0 | 2→36 | 9→0 | 35→0 | 3→36 | - |
| episode017 | FloorPlan5 | True | 0.027027→1.0 | 2→37 | 4→0 | 36→0 | 4→37 | - |
| episode018 | FloorPlan205 | True | 0.025641→1.0 | 2→39 | 5→0 | 38→0 | 2→39 | - |
| episode019 | FloorPlan305 | True | 0.029412→1.0 | 2→34 | 7→0 | 33→0 | 3→34 | - |
| episode020 | FloorPlan405 | True | 0.027778→1.0 | 2→36 | 7→0 | 35→0 | 4→36 | - |

## 解释边界
- ready=false 的 episode 不能作为正式多 episode 探索结论。
- coverage diagnostic 仍只作为上限诊断，不作为 predicted DSG evidence。
