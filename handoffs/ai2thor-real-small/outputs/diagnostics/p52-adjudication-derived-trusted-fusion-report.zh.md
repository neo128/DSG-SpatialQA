# P52 adjudication-derived trusted fusion

- fusion_policy: `adjudication_derived_trusted_graph_or_vlm_fallback`
- calibration_kind: `same_dataset_adjudication_derived`
- not_final_research_claim: `True`
- graph_source_count: 292
- vlm_source_count: 284

该策略把 P50 adjudication 的经验固化为 deterministic gate，但当前仍是 same-dataset calibration；需要在 P54/P55 的 held-out episode 上验证后，才能作为新的独立研究结论。
