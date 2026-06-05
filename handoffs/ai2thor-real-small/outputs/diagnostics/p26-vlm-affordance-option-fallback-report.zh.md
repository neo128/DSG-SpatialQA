# P26 VLM-only Affordance Option Fallback 阶段报告

## 目标

本轮继续提高 VLM-only 对照组。P26 针对 P25 后仍然存在的
`target_not_observed` 失败：当 request bundle 已经给出公开 answer options，
且目标类别与某个公开支撑物存在明确的常识性 affordance 对应时，用该公开 option
补全结构化 `current_location`。

该步骤不调用外部 VLM/LLM，不读取 gold answer，也不读取 gold evidence。它是
`VLM + public answer options + target affordance prior`，不是裸 VLM-only。

## 方法

P26 的适用条件：

- prediction `error` 必须是 `target_not_observed` 或 `relation_not_observed`；
- prediction 不能已有可用 `current_location`；
- request bundle 中必须有目标 label 和公开 answer options；
- 目标类别必须在手工声明的 affordance prior 中；
- 公开 options 中只能唯一命中一个 preferred support；
- 如果多个 preferred supports 同时出现，则跳过，避免过度猜测。

示例 prior：

- `pen/pencil -> desk`
- `mug -> desk/coffeetable/countertop/diningtable`
- `cellphone/laptop -> bed/desk`
- `bread/butterknife -> countertop`
- `bowl -> coffeetable/countertop/desk/diningtable`

## 数据与产物

输入 prediction：

- `inputs/offline-controls/reruns/observation-aware-p4-target60/vlm-p25-text-option-alignment.jsonl`

使用的 request bundle：

- `offline-control-prediction-request-bundle-observation-aware-p4-target60-vlm-p21-ready35-crop-complete.json`

输出 artifacts：

| artifact | path |
| --- | --- |
| P26 prediction | `inputs/offline-controls/reruns/observation-aware-p4-target60/vlm-p26-affordance-option-fallback.jsonl` |
| P26 fallback report | `outputs/diagnostics/vlm-p26-affordance-option-fallback-report.json` |
| P26 semantic eval | `outputs/diagnostics/vlm-semantic-eval-p26-affordance-option-fallback.json` |
| P26 QA eval | `outputs/diagnostics/vlm-qa-eval-p26-affordance-option-fallback.json` |
| P26 vs P25 delta | `outputs/diagnostics/vlm-p26-affordance-option-fallback-vs-p25-text-option-alignment-semantic-delta.json` |
| P26 vs P21 delta | `outputs/diagnostics/vlm-p26-affordance-option-fallback-vs-p21-pending-semantic-delta.json` |
| DSG P22 vs VLM P26 delta | `outputs/diagnostics/dsg-p22-vs-vlm-p26-affordance-option-fallback-semantic-delta.json` |

## 结果

P26 fallback report：

| metric | value |
| --- | ---: |
| applied fallbacks | 12 |
| input predictions | 60 |
| case inputs in retry bundle | 35 |
| missing case inputs | 25 |
| skipped ineligible error | 17 |
| skipped no affordance option | 6 |
| skipped ambiguous affordance option | 0 |

Semantic eval：

| version | semantic match | semantic rate | strict exact |
| --- | ---: | ---: | ---: |
| P21 scoped pending merge | 24 / 60 | 0.400000 | 0 / 60 |
| P24 single-support fallback | 36 / 60 | 0.600000 | 0 / 60 |
| P25 text option alignment | 37 / 60 | 0.616667 | 0 / 60 |
| P26 affordance option fallback | 49 / 60 | 0.816667 | 0 / 60 |

P26 vs P25：

| metric | value |
| --- | ---: |
| semantic count delta | +12 |
| semantic rate delta | +0.200000 |
| paired wins / losses / ties | 12 / 0 / 48 |
| decision | `candidate_improved` |

DSG P22 vs VLM P26：

| method | semantic match | semantic rate |
| --- | ---: | ---: |
| VLM-only P26 | 49 / 60 | 0.816667 |
| coverage-merged DSG / GraphTool P22 | 60 / 60 | 1.000000 |

| metric | value |
| --- | ---: |
| semantic count delta | +11 |
| semantic rate delta | +0.183333 |
| paired wins / losses / ties | 11 / 0 / 49 |
| decision | `candidate_improved` |

## 剩余失败

P26 后剩余 11 个失败，主要分为：

- `chair/table` 歧义：book、laptop、newspaper 的文本只说 `table`，公开 options
  有 `chair/diningtable/floor`，继续盲修风险很高；
- room-level 大件或地面对象：box、basketball、boots 等仍为
  `target_not_observed`；
- ambiguous relation：cloth 在 bathtub 上/内，不能无证据选择；
- crop/input gap：cd 使用的是较早 bundle，缺少 P21 ready35 crop-complete 输入。

这些失败更适合通过真实强 prompt VLM retry 或更好的视觉输入解决，不建议继续用
规则后处理硬修。

## 阶段结论

P26 把 VLM-only semantic baseline 从 37/60 提高到 49/60，较 P21 的 24/60
总提升为 +25。DSG P22 仍为 60/60，领先缩小到 +11。

这个结果让对照组明显更强，也让 DSG 的领先更有说服力。但 P26 已经是
answer-option constrained + affordance prior 的增强 VLM baseline，不应被写成
裸 VLM-only。最终结论仍需要真实强 VLM retry prediction 和 detector-only DSG
readiness 同时通过后再复算。
