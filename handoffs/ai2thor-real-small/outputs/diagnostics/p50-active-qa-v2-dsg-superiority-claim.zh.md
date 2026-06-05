# P50 active QA v2 DSG superiority claim

- claim_allowed: True
- scope: active QA v2 over 5 real AI2-THOR reachable relation-centric NBV episodes, comparing real Qwen VLM-only against real Qwen VLM+DSG adjudication using local predicted DSG/GraphTool candidates
- navigation: 5/5 episode formal ready
- adjudication: ready=True, predictions=576/576
- VLM-only semantic: 141/576 (0.244792)
- VLM+DSG adjudicated semantic: 283/576 (0.491319)
- paired wins/losses/ties: 142/0/434
- sign_test_p_value: 0.0

## 允许的结论
在 5 个真实 AI2-THOR reachable relation-centric NBV episode 的 active QA v2 上，真实 Qwen VLM+DSG adjudication 显著优于真实 Qwen VLM-only。

## 结论边界
- not a general proof across all AI2-THOR scenes
- not a claim that GraphTool-only is an external model
- not a claim about unobserved full-oracle QA targets

## Question Type 分组

| type | cases | VLM | VLM+DSG | delta |
| --- | ---: | ---: | ---: | ---: |
| object_location | 121 | 0 | 37 | 37 |
| situated_egocentric | 200 | 114 | 145 | 31 |
| support_relation | 55 | 0 | 38 | 38 |
| temporal_last_seen | 200 | 27 | 63 | 36 |

## Episode 分组

| episode | cases | VLM | VLM+DSG | wins/losses/ties | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| ai2thor-real-small-episode-001 | 146 | 36 | 68 | 32/0/114 | 0.0 |
| ai2thor-real-small-episode-002 | 82 | 14 | 33 | 19/0/63 | 4e-06 |
| ai2thor-real-small-episode-003 | 106 | 33 | 51 | 18/0/88 | 8e-06 |
| ai2thor-real-small-episode-004 | 87 | 19 | 67 | 48/0/39 | 0.0 |
| ai2thor-real-small-episode-005 | 155 | 39 | 64 | 25/0/130 | 0.0 |
