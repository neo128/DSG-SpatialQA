# P25 VLM-only Text Option Alignment 阶段报告

## 目标

本轮继续优先提高 VLM-only 对照组，而不是继续优化 DSG。P25 只做一个保守的
no-gold 后处理：当 VLM 已经给出可用 `current_location`，且 reasoning/answer
文本唯一提到某个公开 answer option 时，将结构化答案对齐到该 option。

该步骤不调用外部 VLM/LLM，不读取 gold answer，也不读取 gold evidence。

## 方法

P25 的适用条件：

- prediction `error` 必须为空；
- prediction 必须已有可用 `current_location`；
- 当前 destination 不能已经是公开 answer option；
- request bundle 中必须存在同 relation 的非 room answer option；
- VLM 文本只能唯一命中一个公开 option label；
- 普通 `table` 不会被泛化成 `diningtable`，避免过度修正。

典型修正是：VLM 文本中同时出现 “small shelf / shelves”，但结构化答案写成
`sidetable` 时，如果公开 option 中存在 `shelf`，则对齐到 `shelf`。

## 数据与产物

输入 prediction：

- `inputs/offline-controls/reruns/observation-aware-p4-target60/vlm-p24-single-support-option-fallback.jsonl`

使用的 request bundle：

- `offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p21-ready35-crop-complete.json`

输出 artifacts：

| artifact | path |
| --- | --- |
| P25 prediction | `inputs/offline-controls/reruns/observation-aware-p4-target60/vlm-p25-text-option-alignment.jsonl` |
| P25 alignment report | `outputs/diagnostics/vlm-p25-text-option-alignment-report.json` |
| P25 semantic eval | `outputs/diagnostics/vlm-semantic-eval-p25-text-option-alignment.json` |
| P25 QA eval | `outputs/diagnostics/vlm-qa-eval-p25-text-option-alignment.json` |
| P25 vs P24 delta | `outputs/diagnostics/vlm-p25-text-option-alignment-vs-p24-single-support-option-fallback-semantic-delta.json` |
| P25 vs P21 delta | `outputs/diagnostics/vlm-p25-text-option-alignment-vs-p21-pending-semantic-delta.json` |
| DSG P22 vs VLM P25 delta | `outputs/diagnostics/dsg-p22-vs-vlm-p25-text-option-alignment-semantic-delta.json` |

## 结果

P25 alignment report：

| metric | value |
| --- | ---: |
| aligned predictions | 2 |
| input predictions | 60 |
| case inputs in retry bundle | 35 |
| missing case inputs | 25 |
| skipped ineligible error | 18 |
| skipped no text match | 13 |
| skipped existing option location | 2 |

被对齐的 case：

- `bottle_01_54_00_89_02_54`：`sidetable` 对齐到 `shelf`，该 case 变为正确；
- `newspaper_02_76_00_68_01_18`：文本对齐到 `diningtable`，但 gold 是 `chair`，未变为正确。

Semantic eval：

| version | semantic match | semantic rate | strict exact |
| --- | ---: | ---: | ---: |
| P21 scoped pending merge | 24 / 60 | 0.400000 | 0 / 60 |
| P24 single-support fallback | 36 / 60 | 0.600000 | 0 / 60 |
| P25 text option alignment | 37 / 60 | 0.616667 | 0 / 60 |

P25 vs P24：

| metric | value |
| --- | ---: |
| semantic count delta | +1 |
| semantic rate delta | +0.016667 |
| paired wins / losses / ties | 1 / 0 / 59 |
| decision | `candidate_improved` |

DSG P22 vs VLM P25：

| method | semantic match | semantic rate |
| --- | ---: | ---: |
| VLM-only P25 | 37 / 60 | 0.616667 |
| coverage-merged DSG / GraphTool P22 | 60 / 60 | 1.000000 |

| metric | value |
| --- | ---: |
| semantic count delta | +23 |
| semantic rate delta | +0.383333 |
| paired wins / losses / ties | 23 / 0 / 37 |
| decision | `candidate_improved` |

## 阶段结论

P25 将 VLM-only semantic baseline 从 36/60 小幅提高到 37/60。提升幅度有限，
但规则是可审计、no-gold、确定性的，并且没有造成 paired loss。

当前更强 VLM-only 后处理链路为：

`P21 pending merge -> P22 room fallback -> P23 room-level target fallback -> P24 single-support fallback -> P25 text option alignment`

它把 VLM-only 从 24/60 提高到 37/60。DSG P22 仍为 60/60，领先 23 个 case。
不过 P25 仍是 answer-option constrained 后处理，不等同于真实重新跑强 prompt 的
VLM-only。下一步需要真实 VLM retry prediction 返回后复算，尤其是 18 个
`target_not_observed` case。
