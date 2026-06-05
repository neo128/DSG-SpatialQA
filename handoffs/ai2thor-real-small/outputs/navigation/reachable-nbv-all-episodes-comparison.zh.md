# reachable NBV 多 episode 协议对比

## 总结
- episode_count: 5
- formal_protocol_ready_episode_count: 4
- all_episodes_formal_protocol_ready: False

## Episode 表

| episode | scene | ready | same_frame fixed→NBV | evidence fixed→NBV | missing_support fixed→NBV | missing_relation fixed→NBV | GraphTool semantic fixed→NBV | failed_checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| episode001 | FloorPlan1 | True | 0.083333→0.75 | 2→12 | 1→0 | 11→3 | 3→9 | - |
| episode002 | FloorPlan201 | True | 0.083333→0.416667 | 2→10 | 1→0 | 11→7 | 2→5 | - |
| episode003 | FloorPlan301 | True | 0.083333→0.5 | 2→12 | 1→0 | 11→6 | 0→6 | - |
| episode004 | FloorPlan401 | True | 0.25→0.5 | 6→11 | 4→1 | 9→6 | 1→5 | - |
| episode005 | FloorPlan2 | False | 0.166667→0.0 | 4→0 | 3→2 | 10→12 | 1→0 | target_support_same_frame_rate_gt_fixed,evidence_observable_qa_count_gte_fixed,missing_relation_count_lt_fixed,graphtool_semantic_match_gte_fixed |

## 解释边界
- ready=false 的 episode 不能作为正式多 episode 探索结论。
- coverage diagnostic 仍只作为上限诊断，不作为 predicted DSG evidence。
