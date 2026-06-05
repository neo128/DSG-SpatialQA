# p5 语义归一化阶段报告

## 目标

本阶段目标是排查 `observation-aware p4 target60` 上 DSG candidate 语义评估异常偏低的问题，并在不调用外部 VLM/LLM、不修改真实预测文件的前提下，校准本地语义评估口径。

## 根因

原语义评估能把 VLM 文本答案 `on the countertop` 归一化为 `relation=ON, destination_label=countertop`，但对 GraphTool 的结构化答案：

```json
{"current_location": {"relation": "ON", "dst": "countertop_00_08_01_15_00_00"}}
```

只做普通文本 canonicalization，结果变成 `countertop000801150000`，无法和 gold 的 `countertop` 匹配。因此部分正确的 DSG 结构化位置答案会被判为 `destination_mismatch`。

## 修改

在 `src/dsg_spatialqa_lab/eval/vlm_calibration.py` 中，prediction 的结构化 destination 改用和 gold 一致的 graph/object id label 提取逻辑。新增测试覆盖 GraphTool 结构化 `current_location.dst` 和 gold `dst` 的语义等价。

## 本地验证

- `python -m pytest tests/test_vlm_calibration.py -q`
- `scripts/eval_vlm_calibration.py --validate-semantic-eval-report ...` 三个 p5 semantic eval report 均 valid。
- 两个 p5 semantic delta report 均通过 `validate_vlm_semantic_eval_delta_report`。

## p5 指标

同一 `observation-aware-p4-target60` 60 条 object-location QA 上：

| 系统 | Semantic Match | Rate |
| --- | ---: | ---: |
| DSG GraphTool candidate | 44/60 | 0.733333 |
| VLM-only qwen3.7 p4 | 17/60 | 0.283333 |
| Multi-frame VLM qwen3.7 window4 p4 | 25/60 | 0.416667 |

Delta：

| 对比 | Wins | Losses | Ties | Rate Delta |
| --- | ---: | ---: | ---: | ---: |
| DSG vs VLM-only | 28 | 1 | 31 | +0.450000 |
| DSG vs Multi-frame VLM | 26 | 7 | 27 | +0.316667 |

## Artifact

- `dsg-candidate-semantic-eval-p5-canonicalized.json`
- `vlm-semantic-eval-p5-canonicalized-independent.json`
- `multi-frame-vlm-semantic-eval-p5-canonicalized-independent.json`
- `dsg-candidate-semantic-delta-vs-vlm-p5-canonicalized.json`
- `dsg-candidate-semantic-delta-vs-multi-frame-vlm-p5-canonicalized.json`

## 结论

p5 证明：在已构建的 observation-aware p4 target60 slice 上，DSG GraphTool 的结构化位置答案并非 0/60；修正语义归一化后，DSG candidate 明显高于当前 VLM-only 和 multi-frame VLM baseline。

但这仍是阶段性诊断结论，不是最终 real research-ready claim。最终声明还需要通过真实 readiness gate，尤其是 independent detector/RGB-D evidence 不能来自 simulator metadata，并且需要在同一输入增强策略下重跑 VLM-only P1 support+crop baseline。

## 下一步

1. 使用 `vlm-p1-support-crop` request bundle 重跑 VLM-only，目标提高并校准强 baseline。
2. 使用 support-aware detector handoff 重跑 external detector/RGB-D，目标让 predicted DSG evidence 通过 detector/rgb/depth gate。
3. 在同一 QA slice 上重跑 final semantic delta、QA eval、graph eval、error attribution 和 readiness。
